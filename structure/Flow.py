
# -*- encoding: utf-8 -*-
'''
@File    :   Flow.py
@Time    :   2022/11/26 19:42:45
@Author  :   Hannah 
@Version :   1.0
@Contact :   1196033301@qq.com
@Desc    :   Flow class, PrintFlow class, FlowManager class
'''

# here put the import lib


# sys import
from typing import *
from colorama import Fore
import time
import copy
import os,sys

# 本级目录模块的import
this_filepath = __file__ # 此文件的路径: exp_Robust/utils/MyColor.py
this_dirpath = os.path.dirname(this_filepath) # 此文件所在文件夹/目录的路径: exp_Robust/utils
sys.path.append(this_dirpath) 
# e.g. import
from Packet import *
from DataTool import *

# 上级目录模块的import: 将上级目录天加入环境变量，以便引入其他模块
parent_dirpath = os.path.join(this_dirpath, "..") # 加上..表示当前路径的上层目录:exp_Robust/
sys.path.append(parent_dirpath)  # 将上级目录添加到sys.path中，使得上级目录的模块可以被import
from config import *
from utils.MyColor import *
from utils.MyMongoDB import *


class Flow:

    def __init__(self) -> None:  
      
        self.fivetuple: List[str] = []
        self.belongfile = ""
        self.lst_packets:List[Packet] = [] # packet更新 ，则相应的流也会更新
        # for query
        
        self.label = "" # black or white
        self.type_dataset = ""  # train or test
        self.type_data = "Flow" # fixed
        self.desc = {} # 用于描述流的一些信息
        
    def get_BagID(self)->tuple:  # threetuple
        # src_ip dst_ip filename
        return (self.fivetuple[0],self.fivetuple[1],self.belongfile)
        
    def get_BagID_reversed(self)->tuple:
        # dst_ip src_ip filename
        return (self.fivetuple[1],self.fivetuple[0],self.belongfile)

    def get_fivetuple(self)->tuple:
        return tuple(self.fivetuple)

    def get_FlowID(self) -> tuple:
        return self.get_fivetuple()

    def get_fivetuple_reversed(self)->tuple:
        return (self.fivetuple[1],self.fivetuple[0],self.fivetuple[3],self.fivetuple[2],self.fivetuple[4])

    def get_FlowID_reversed(self) -> tuple: 
        return self.get_fivetuple_reversed()

    def create(self,firstpacket: Packet, belongfile: str, label: str="?", type_dataset: str="?"):
        self.fivetuple = firstpacket.fivetuple
        self.belongfile = belongfile
        self.lst_packets.append(firstpacket)
        self.label = label
        self.type_dataset = type_dataset
        self.type_data = "Flow"
        self.desc = {}
        return self

    def append(self, packet: Packet) -> bool:
        # 添加一个包
        
        if packet.get_FlowID()!= self.get_FlowID() and packet.get_FlowID_reversed() != self.get_FlowID():
            return False

        self.lst_packets.append(packet)
        return True

    def toDict(self)->dict:
        dict_flow =copy.deepcopy(self.__dict__)  # 必须深复制，否则会改变原来的值
        dict_flow['lst_packets'] = []
        if len(self.lst_packets)>0:
            for packet in self.lst_packets:
                dict_flow['lst_packets'].append(packet.toDict()) 
        return dict_flow

    def fromDict(self, dict_flow):
        self.__dict__ = copy.deepcopy(dict_flow) # 必须深复制，否则会改变原来的值
        # 构造lst_packets，元素为一个Packet对象
        self.lst_packets = [] 
        for packet in dict_flow['lst_packets']:  # 不能写列表生成式
            temp_packet = Packet()
            temp_packet.fromDict(packet)
            self.lst_packets.append(temp_packet)
        # 结束一个流: 对包按照时间戳排序
        self.finish()
        return self

    def finish(self):
        # 用于结束一个流, 需要对包按照时间戳排序
        self.lst_packets.sort(key=lambda x: x.timestamp)

    def set_label(self, label):
        self.label = label
        
    def set_type_dataset(self, type_dataset):
        self.type_dataset = type_dataset

    def add_desc(self, add_desckey, add_descvalue):
        self.desc[add_desckey] = add_descvalue

    def cnt_duration(self) -> float:
        self.finish()
        start_packet = self.lst_packets[0]
        end_packet = self.lst_packets[-1]
        duration = end_packet.timestamp - start_packet.timestamp
        return duration

# 查看Flow类 
class PrintFlow(PrintTool):
    def __init__(self, print_color: PrintColor = PrintColor()) -> object:
        # 设置输出颜色的模式
        # example
        PrintTool.__init__(self, print_color)

    def print_abcinfo(self,flow: Flow):
        src = flow.fivetuple[0]
        dst = flow.fivetuple[1]
        src_p = flow.fivetuple[2]
        dst_p = flow.fivetuple[3]
        protocol = flow.fivetuple[4]
        self.print_NI_info(f"flow:\t|{src}:{src_p} > {dst}:{dst_p} ,{protocol}\t|file:{flow.belongfile}")

        # 输出一共多少个数据包
        self.print_info(f"~~total packets: {len(flow.lst_packets)}")
        # 输出label
        if flow.label == "Unknown":

            self.print_info(f"~~label: {Fore.MAGENTA}{flow.label}{Fore.RESET}")
        else:
            self.print_info(f"~~label: {flow.label}")
        # 输出流持续时间
        self.print_info(f"~~duration: {flow.lst_packets[-1].timestamp - flow.lst_packets[0].timestamp}\n")
        if flow.desc != {}:
            self.print_info(f"~~desc: {Fore.MAGENTA}{flow.desc}{Fore.RESET}")

    def print_details(self,flow: Flow):
        src = flow.fivetuple[0]
        dst = flow.fivetuple[1]
        src_p = flow.fivetuple[2]
        dst_p = flow.fivetuple[3]
        protocol = flow.fivetuple[4]
        self.print_NI_info(f"flow:\t|{src}:{src_p} > {dst}:{dst_p} ,{protocol}\t|file:{flow.belongfile}\t|Label:{flow.label}")
        for i, packet in enumerate(flow.lst_packets):
            # PNo: 第几个包
            print(f"\tPNo.{i:>2}:   l: {packet.length:3},  d: { packet.direction:2},  t: {packet.timestamp}")
            if  packet.cmp_origin()==False:
                self.print_attention(f"\t       origin: {list(packet.origin.values())}")


        # 输出一共多少个数据包
        self.print_info(f"~~total packets: {len(flow.lst_packets)}")
        # 输出label
        if flow.label == "Unknown":

            self.print_info(f"~~label: {Fore.MAGENTA}{flow.label}{Fore.RESET}")
        else:
            self.print_info(f"~~label: {flow.label}")
        # 输出流持续时间
        self.print_info(f"~~duration: {flow.lst_packets[-1].timestamp - flow.lst_packets[0].timestamp}\n")
        if flow.desc!= {}:
            self.print_info(f"~~desc: {Fore.MAGENTA}{flow.desc}{Fore.RESET}")

# flow <-> database
class FlowManager(DataManager):
    def __init__(self,lst_flow:List[Flow]=[], name: str="flow&db", logger: MyLogger=None) -> None:
        DataManager.__init__(self, logger, print_name=f"|FlwMngr「{name}」|")
        self.lst_flows: List[Flow] = lst_flow
        
    def create_from_lst_flow(self, lst_flow: List[Flow]):
        self.lst_flows = lst_flow
        return self

    def get_datas(self)->List[Flow]:
        return self.lst_flows

    def clear(self):
        self.lst_flows = []

    def append(self, flow: Flow):
        self.lst_flows.append(flow)

    def extend(self, lst_flow: List[Flow]):
        self.lst_flows.extend(lst_flow)

    def cnt(self) -> int:
        return len(self.lst_flows)

    def count_allflows(self) -> int:
        return len(self.lst_flows)

    def save_to_mongodb(self, mydb: MyMongoDB):
        # 保存到mongodb
        lst_dict_flow:List[Dict] = []
        for flow in self.lst_flows:
            dict_flow = flow.toDict()
            lst_dict_flow.append(dict_flow)
        try:
            mydb.insert(lst_dict_flow)
            self.print_right(f"{self.print_name}: save -> {mydb.print_name}: {len(lst_dict_flow)} flows saved.")

        except Exception as e:
            self.print_wrong("save flow wrong")
        self.print_info(f"\tINFO: {mydb.print_name} has {mydb.cnt_all()} flows in total")
    
    # 从数据库中读取数据
    def addread_from_mongodb(self, mydb: MyMongoDB, query: Dict = {}):
        # 如 query={"datatype":"black","dataset_type":"train"} 训练集的黑样本
        lst_dict_flow = mydb.find(query)
        num = len(lst_dict_flow)
        for dict_flow in lst_dict_flow:
            if isinstance_dict(dict_flow) != OBJ_DctFlw:
                num -= 1
                continue
            flow = Flow()
            flow.fromDict(dict_flow)
            self.lst_flows.append(flow)
        self.print_right(f"{self.print_name}: addread <- {mydb.print_name}: {num} flows read.")
        self.print_info(f"\tINFO: has {len(self.count_allflows())} flows in total after addread")

    def create_from_mongodb(self, mydb: MyMongoDB, query: Dict = {}):
        self.lst_flows = []
        lst_dict_flow = mydb.find(query)
        num = len(lst_dict_flow)
        for dict_flow in lst_dict_flow:
            if isinstance_dict(dict_flow) != OBJ_DctFlw:
                num -= 1
                continue
            flow = Flow()
            flow.fromDict(dict_flow)
            self.lst_flows.append(flow)
        self.print_right(f"{self.print_name}: create <- {mydb.print_name}: {len(self.lst_flows)} flows read.")
        return self
    

if __name__ == "__main__":
    # sample
    # 创建packet
    packet1 = Packet()
    packet1.create(["A_ip","B_ip","A_port","B_port","t"], 100, 5, int(time.time()))
    packet2 = Packet()
    packet2.create(["A_ip","B_ip","A_port","B_port","t"], 200, 7, int(time.time()+2))
    packet2.update(length=1000)
    # 创建flow
    flow1 = Flow()
    flow2 = Flow()
    flow1.create(packet1,belongfile="file1")
    flow1.append(packet2)
    flow2.create(packet2,belongfile="file2")

    # 连接数据库
    wl = MyMongoDB('test',"test_flow")
    # 查看前n条数据
    wl.print_front()
    # 全部删除
    wl.delete_all()
    # 插入数据：首先将flow转换为dict，然后插入数据库
    wl.insert([flow1.toDict(),flow2.toDict()])
    # 查看插入结果
    wl.print_front()
    # 查找所有数据
    lst_dict_flow = wl.find()
    for i in lst_dict_flow:
        flowtemp = Flow()
        flowtemp.fromDict(i)
        PrintFlow().print_details(flowtemp)
    