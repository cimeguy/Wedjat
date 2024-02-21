# -*- encoding: utf-8 -*-
'''
@File    :   PreProcess.py
@Time    :   2022/11/09 18:57:19
@Author  :   Hannah 
@Version :   1.0
@Contact :   1196033301@qq.com
@Desc    :   pcap file -> Packet -> List[Flow]
            # 1. 读取pcap文件
            # 2. 将pcap文件转换为List[Flow]
            # 3. 将List[Flow]存储到数据库中
            # 4. 从数据库中读取List[Flow]
            # 不设定数据集的类型 train/ test/ val
'''
# sys import
from tqdm import tqdm
import dpkt
from dpkt.ip import get_ip_proto_name
import socket

# this dir import
from config import *
from ABCClass import Operation


# sub dir import
# # utils
from utils.MyMongoDB import *
from utils.MyLogger import *
from utils.MyColor import *
# # structure
from structure.DataTool import *
from structure.Flow import *
from structure.Packet import *




def print_timestamp(timestamp):
    print(str(datetime.datetime.utcfromtimestamp(timestamp)))


class ReadFile(Operation):
    def __init__(self, logger: MyLogger=None, 
        filetype: List[str]= ["pcap", "dump", "cap"]) -> None:

        Operation.__init__(self,logger, print_name="｜ReadFile｜")
        self.filetype = filetype # 需要哪些文件类型
        
    def _get_allfile_in_dir(self,dirpath):
        # find all file in <dirpath>, whose type is in <filetype> 
        lst = os.listdir(dirpath)
        pathlst = []
        for name in lst: # name is file_name
            path = os.path.join(dirpath, name)
            suffix = name.split(".")[-1]
            if os.path.isdir(path):
                extendlst = self._get_allfile_in_dir(path)
                pathlst.extend(extendlst)
            elif suffix in self.filetype:
                pathlst.append(path)
            else:
                pass
        return pathlst
    
    def get_allfile(self,dirpath: str)->List[str]:
        if not os.path.exists(dirpath):
            print(self.print_wrong(f"pcap dir {dirpath} is not exist"))
            return []
        lst_filepath = self._get_allfile_in_dir(dirpath)
        self.lst_filepath = lst_filepath
        # print
        self.print_info(f"\tINFO: find {len(lst_filepath)} files")
        self.print_info(f"\t      dirpath = {dirpath}")
         # log
        self.logger.info(f"{self.op_name}: find {len(lst_filepath)} files")
        return lst_filepath




class Processor(Operation):

    def __init__(self, logger: MyLogger=None, auto_label: bool=True) -> None:

        self.auto_label = auto_label # 自动根据文件名获得data label
        Operation.__init__(self,logger,print_name="｜Processor「pcap2flow」｜")
        if self.auto_label:
            self.print_attention("\t根据文件名获得样本的label")
        else:
            self.print_attention("\t不根据文件名获得样本的label")

    def set_auto_label(self, auto_label: bool=True)->None:
        self.auto_label = auto_label

    def _inet_to_str(self,inet):
        """Convert inet object to a string

            Args:
                inet (inet struct): inet network address
            Returns:
                str: Printable/readable IP address
        """
        # First try ipv4 and then ipv6
        # return socket.inet_ntop(socket.AF_INET, inet)# if len(inet) == 4 else socket.inet_ntop(socket.AF_INET6, inet)
        try: # ipv4
            return socket.inet_ntop(socket.AF_INET, inet)
        except ValueError: # ipv6
            return socket.inet_ntop(socket.AF_INET6, inet)
       

    def process_onefile(self, filepath:str, label)->List[Flow]:
        dict_flows_onefile: Dict[tuple, Flow] ={} # 一个文件中的所有流: Dict[five_tuple, Flow]
        lst_flows_onefile = list(dict_flows_onefile.values())
        # 读取pcap文件
        with open(filepath, 'rb') as pcap_file:
            # each pcap has many flows
            try:
                reader = dpkt.pcap.Reader(pcap_file)
            except ValueError as e_pcap:
                
                try:
                    pcap_file.seek(0, os.SEEK_SET)
                    reader = dpkt.pcapng.Reader(pcap_file)
                except ValueError as e_pcapng:
                    self.print_wrong("不是PCAP or PCAPng:1")
                    self.logger.error(f"不是PCAP or PCAPng:{filepath}")
                    return []
                except BaseException:
                    self.print_wrong("other BaseException2")
                    self.logger.error(f"File raise other BaseException:{filepath}")
                return []
            except dpkt.dpkt.NeedData:
                self.print_wrong(f"dpkt.dpkt.NeedData:{filepath}")
                self.logger.error(f"File raise dpkt.dpkt.NeedData:{filepath}")
                return []
            # 捕获NeedData异常
            for ts, buf in reader:
                
                try: 
                    eth = dpkt.ethernet.Ethernet(buf)
                except Exception(e):
                    # logging.error("Exception(e):"+PcapPath)
                    continue
                if not isinstance(eth.data, dpkt.ip.IP):
                    continue
                ip = eth.data
                if not isinstance(ip.data, dpkt.tcp.TCP):
                    continue
                tcp = ip.data
                extra_info= ""



                # dpkt.ssl.tls_multi_factory(tcp.data)
                
                # if isinstance(tcp, dpkt.ssl.SSL3) :
                #     print(f'SSL3 packet found.[{filepath}]')
                if isinstance(tcp.data, dpkt.ssl.TLS) :
                    tls = dpkt.ssl.TLS(tcp.data)
                    print(f'TLS packet found.[{filepath}]')
                    extra_info=""
                else:
                    # print(f'This packet is not encrypted with TLS protocol.[{filepath}]')
                    extra_info = "nonTLS"
                protocol = get_ip_proto_name(ip.p)
                
                src_addr = self._inet_to_str(ip.src)
                
                dst_addr = self._inet_to_str(ip.dst)
                src_port = str(tcp.sport)
                # protocol = get_ip_proto_name(tcp.p)
                dst_port = str(tcp.dport)
                packet=Packet()
                fivetulpe = (src_addr, dst_addr, src_port, dst_port, protocol)
                reverse_fivetulpe = (dst_addr, src_addr, dst_port,src_port, protocol)
                packet.create(fivetuple=list(fivetulpe), length=len(buf), direction=0, timestamp=ts)
                
                packet.origin["extra_info"] = extra_info
                if fivetulpe in dict_flows_onefile.keys():
                    packet.direction = DIR_src2dst
                    packet.origin["direction"] = DIR_src2dst
                    dict_flows_onefile[fivetulpe].append(packet)
                elif reverse_fivetulpe in dict_flows_onefile.keys():
                    
                    packet.direction = DIR_dst2src
                    packet.origin["direction"] = DIR_dst2src
                    dict_flows_onefile[reverse_fivetulpe].append(packet)
                else:
                    packet.direction = DIR_src2dst
                    packet.origin["direction"] = DIR_src2dst
                    flow = Flow()
                    flow.create(firstpacket=packet, belongfile=filepath, label=label)
                    dict_flows_onefile[fivetulpe] = flow        
          
        lst_flows_onefile = list(dict_flows_onefile.values())
        for flow in lst_flows_onefile:
            flow.finish()
        return lst_flows_onefile
    def get_label_from_filename(self, filename:str)->str:
        # 从文件名中获得标签
        if ("BLACK" in filename and "WHITE"in filename) \
            or ("BLACK" in filename and "white" in filename)\
                or ("black" in filename and "WHITE" in filename)\
                    or ("black" in filename and "white" in filename):
            return UNKNOWN
       
        if "BLACK" in filename or "black" in filename:
            return BLACK
        elif "WHITE" in filename or "white" in filename:
            return WHITE
        else:
            return UNKNOWN

    def process_allfile(self, lst_filepath:List[str], label=UNKNOWN)->List[Flow]:
        lst_flow = []
        for filepath in tqdm(lst_filepath, desc="process_allfile"):
            if self.auto_label:
                label = self.get_label_from_filename(filepath)
            lst_flow.extend(self.process_onefile(filepath, label))
        
        self.print_info(f"\t this set has {len(lst_flow)} flows in total")
        return lst_flow



def main2():
    a = Flow()
    p1 = Packet().create(test1_5tuple,100,DIR_src2dst,1.0)
    p2 = Packet().create(test1_5tuple,200,DIR_dst2src,2.0)
    p3 = Packet().create(test1_5tuple,300,DIR_src2dst,3.0)
    a.create(p1,belongfile="test")
    a.append(p3)
    p3.update(length=400) # 修改后 相应的流也会修改
    p2.update(timestamp=4.0)
    a.append(p2)
    a.finish()
    pa = PrintFlow()
    pa.print_abcinfo(a)
    pa.print_details(a)


def test(log_filename="test.log"):
    dataset_dirname = "mytest"
    blackset_dirname = "test_white" # in dir[dataset_name]
    blackset_dirpath = os.path.join(dataset_dir, dataset_dirname, blackset_dirname)
    # 创建日志记录器
    mylogger = MyLogger(log_filename = log_filename ,filemode="w")
    # 根据dataset和blackset连接数据库
    mydb = MyMongoDB(dataset_dirname, blackset_dirname)
    # 创建文件夹处理器： 得到文件夹下的所有文件的路径
    lst_files = ReadFile(logger=mylogger).get_allfile(dirpath=blackset_dirpath)
    # 创建流处理器： 读取文件，创建流
    lst_flows = Processor(logger=mylogger, auto_label=True).process_allfile(lst_files)
    # 创建流查看器
    seeflow = PrintFlow()
    # 创建流存储器
    flowman = FlowManager(logger=mylogger,name="流处理1").create_from_lst_flow(lst_flows)
    # 清空数据库

    mydb.clear()
    # 存储流
    flowman.save_to_mongodb(mydb=mydb)
    flowman.create_from_mongodb(mydb=mydb)
    for flow in flowman.lst_flows[:10]:
        
        if flow.fivetuple[0] == "192.168.56.107" and flow.fivetuple[2] == "1039":
        # 查看一条流
            seeflow.print_details(flow)
    flowman.log_info("流处理1: 包含{}个流".format(flowman.count_allflows()))



def test(log_filename="test.log"):
    dataset_dirname = "mytest"
    blackset_dirname = "test_black" # in dir[dataset_name]
    blackset_dirpath = os.path.join(dataset_dir, dataset_dirname, blackset_dirname)
    # 创建日志记录器
    mylogger = MyLogger(log_filename = log_filename ,filemode="w")
    # 根据dataset和blackset连接数据库
    mydb = MyMongoDB(dataset_dirname, blackset_dirname+"_flow_preprocess")
    # 创建文件夹处理器： 得到文件夹下的所有文件的路径

    lst_files = ReadFile(logger=mylogger).get_allfile(dirpath=blackset_dirpath)
    # 创建流处理器： 读取文件，创建流
    lst_flows = Processor(logger=mylogger, auto_label=True).process_allfile(lst_files)
    # 创建流查看器
    seeflow = PrintFlow()
    # 创建流数据集
    flowman = FlowManager(logger=mylogger,name="流处理1").create_from_lst_flow(lst_flows)

    
    # 清空数据库

    mydb.clear()
    # 存储流
    flowman.save_to_mongodb(mydb=mydb)
    flowman.create_from_mongodb(mydb=mydb)

    packets_from_flows = []
    for flow in flowman.lst_flows[:10]:
        
        packets_from_flows.extend(flow.lst_packets)
        if flow.fivetuple[0] == "192.168.56.107" and flow.fivetuple[2] == "1039":
        # 查看一条流
            seeflow.print_details(flow)
            for p in flow.lst_packets:
                print(p.fivetuple)
            print(flow.type_data)
            print(flow.type_dataset)
    flowman.log_info("流处理1: 包含{}个流".format(flowman.count_allflows()))
    pktman = PacketsManager(logger=mylogger,name="包处理1").create_from_lst_packets(packets_from_flows)
    

def test_Private(log_filename="test.log"):
    dataset_dirname = "Private"# database name
    blackset_dirname = "black_21" # collection name in dir[dataset_name]
    blackset_dirpath = os.path.join(dataset_dir, dataset_dirname, blackset_dirname)
    # 创建日志记录器
    mylogger = MyLogger(log_filename = log_filename ,filemode="w")
    # 根据dataset和blackset连接数据库
    mydb = MyMongoDB(dataset_dirname, blackset_dirname+"_flow_preprocess")
    # 创建文件夹处理器： 得到文件夹下的所有文件的路径
    
    lst_files = ReadFile(logger=mylogger).get_allfile(dirpath=blackset_dirpath)
    # 创建流处理器： 读取文件，创建流
    lst_flows = Processor(logger=mylogger, auto_label=True).process_allfile(lst_files)
    # 创建流查看器
    seeflow = PrintFlow()
    # 创建流数据集
    flowman = FlowManager(logger=mylogger,name="流处理1").create_from_lst_flow(lst_flows)

    
    # 清空数据库

    mydb.clear()
    # 存储流
    flowman.save_to_mongodb(mydb=mydb)
    flowman.create_from_mongodb(mydb=mydb)

    packets_from_flows = []
    for flow in flowman.lst_flows[:10]:
        
        packets_from_flows.extend(flow.lst_packets)
        if flow.fivetuple[0] == "192.168.56.107" and flow.fivetuple[2] == "1039":
        # 查看一条流
            seeflow.print_details(flow)
            for p in flow.lst_packets:
                print(p.fivetuple)
            print(flow.type_data)
            print(flow.type_dataset)
    flowman.log_info("流处理1: 包含{}个流".format(flowman.count_allflows()))
    pktman = PacketsManager(logger=mylogger,name="包处理1").create_from_lst_packets(packets_from_flows)
    
# main
if __name__ == "__main__":

    test_Private()
    # test()

    
    
   
    





