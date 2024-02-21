# -*- encoding: utf-8 -*-
'''
@File    :   Packet.py
@Time    :   2022/11/21 05:42:01
@Author  :   Hannah 
@Version :   1.0
@Contact :   1196033301@qq.com
@Desc    :   Packet-based 调试完成
'''

# 系统模块的import
import os,sys
import time
import copy
from typing import *
from colorama import Fore
from tqdm import tqdm
# 本级目录模块的import
this_filepath = __file__ # 此文件的路径: exp_Robust/utils/MyColor.py
this_dirpath = os.path.dirname(this_filepath) # 此文件所在文件夹/目录的路径: exp_Robust/utils
sys.path.append(this_dirpath) 
# e.g. import
from DataTool import isinstance_dict, OBJ_DctPkt

# 上级目录模块的import: 
# # 将上级目录先加入环境变量，以便引入其他模块
parent_dirpath = os.path.join(this_dirpath, "..") # 加上..表示当前路径的上层目录:exp_Robust/
sys.path.append(parent_dirpath)  # 将上级目录添加到sys.path中，使得上级目录的模块可以被import
# # e.g. import
from config import *
from utils.MyColor import *
from utils.MyMongoDB import *
from ABCClass import *



# 数据类
class Packet:
    '''
    packet in same flow has same 5-tuple and same belongfile
    
    '''
    def __init__(self)->None:
        self.fivetuple: List = [] # 5-tuple
        self.length: int = 0
        self.direction: int = 0
        self.timestamp: float = 0

        # 加入type？
        # self.label = "" # black or white
        # self.type_dataset = ""  # train or test
        # self.type_data = "Packet" # fixed

        self.desc: Dict = {}  # for evasion 
        self.origin: Dict = {} # origin data from create, is not changed
        # self.label = ""
        # self.dataset_type = ""
        # self.datatype = "Packet"
    def get_data(self) -> list:
        return [self.length, self.direction, self.timestamp]

    def get_FlowID(self)->tuple:    
        return tuple(self.fivetuple)

    def get_FlowID_reversed(self) -> tuple:
        return (self.fivetuple[1],self.fivetuple[0],self.fivetuple[3],self.fivetuple[2],self.fivetuple[4])
    def get_fivetuple(self) ->tuple:
        return tuple(self.fivetuple)
    def get_fivetuple_reversed(self) -> tuple:
        return (self.fivetuple[1],self.fivetuple[0],self.fivetuple[3],self.fivetuple[2],self.fivetuple[4])

    def create(self, fivetuple: List, length: int, direction: int, timestamp: float):
        self.fivetuple = fivetuple
        self.length = length
        self.direction = direction
        self.timestamp = timestamp
        # origin: 用于记录原始的数据，用于后续的数据处理，后续不再修改
        self.origin = {"length": length, "direction": direction, "timestamp": timestamp}
        return self

    def update(self, length: int=None, direction: int=None, timestamp: float=None):
        # this is for evasion attack
        if length is not None:
            self.length = length

        if direction is not None:
            self.direction = direction
        if timestamp is not None:
            self.timestamp = timestamp

    def print(self):
        
        
        print(f"\tl:{self.length:3}, d:{self.direction:2}, t:{self.timestamp:<20}",end=" ")
        # print(f"{s:<50}", end="") # 一共占30个宽度
        print(f"5-tuple: {self.fivetuple}")
        
        # print("\t5-tuple: {}, l: {}, d: {}, t: {}". \
            # format(self.fivetuple, self.length, self.direction, self.timestamp))
        if self.cmp_origin()==False:
            print("\t",end="")
        if self.origin["length"] != self.length:
            # line2
            # = "origin: l: {}, d: {}, t: {}". \
            print(f"{Fore.MAGENTA}o_l:{self.origin['length']:<6}{Fore.RESET}, ",end="")
            # print("o_l: {}".format(self.origin["length"]), end = " ")
        if self.origin["direction"] != self.direction:
            print(f"{Fore.MAGENTA}o_d:{self.origin['direction']:<2}{Fore.RESET}, ",end="")
        if self.origin["timestamp"] != self.timestamp:
            print(f"{Fore.MAGENTA}o_d:{self.origin['timestamp']:<14}{Fore.RESET}",end="")
        if self.desc!={}:
            print(f"{Fore.MAGENTA}o_e:{self.desc:<14}{Fore.RESET}",end="")
        print()
    

    def toDict(self) -> Dict:
        dict_packet = copy.deepcopy(self.__dict__)
        return dict_packet

    def fromDict(self, dict_packet:Dict)->None:
        self.__dict__ =copy.deepcopy(dict_packet)

    def cmp_origin(self) -> bool: # 比较是否与origin不同 : True 相同
        if  self.length == self.origin["length"] and \
            self.direction == self.origin["direction"] and \
            self.timestamp == self.origin["timestamp"] and \
            self.desc=={}:
            return True
        else:
            return False

    def compare_with_origin(self)->bool: #  = cmp_origin
        return self.cmp_origin()

    def get_data(self) -> List: # purify data
        return [self.length, self.direction, self.timestamp]

    def is_brother(self, packet: "Packet") -> bool:
        if self.get_FlowID() == packet.get_FlowID():
            return True
        elif self.get_FlowID_reversed() == packet.get_FlowID():
            return True
        else:
            return False



# 操作类：管理List[Packet]<->数据库
class PacketsManager(DataManager):  

    def __init__(self,lst_packet: List[Packet]=[], name: str="Pkt&db",logger: MyLogger=None) -> None:
        # 操作名称
        DataManager.__init__(self, logger, print_name=f"|PktMngr「{name}」|" )
        
        # lst of Packets
        self.lst_packets: List[Packet] = lst_packet
        self.op_name = name
    
        self.print_info(f"\tINFO: has {len(self.lst_packets)} packets in total")
        
    def create_from_lst_packets(self, lst_packets: List[Packet]):
        self.lst_packets = lst_packets
        self.print_info(f"\tINFO: has {len(self.lst_packets)} packets in total")

    def clear(self):
        self.lst_packets = []
    def append(self, packet: Packet):
        self.lst_packets.append(packet)

    def extend(self, lst_packets: List[Packet]):
        self.lst_packets.extend(lst_packets)  

    def print_datasize(self):
        self.print_info(f"{self.print_name} has {len(self.lst_packets)} packets in total")
    
    def cnt(self) -> int:
        return len(self.lst_packets)

    def get_datas(self) -> List[Packet]:
        return self.lst_packets   

    def create_from_mongodb(self, mydb: MyMongoDB, query: Dict = {}):
        # 会覆盖原先内存中的数据lst_packets 
        # query指定数据库满足该条件的数据
        self.lst_packets = [] # 清空
        lst_dict_packet = mydb.find(query)
        num = len(lst_dict_packet)
        for dict_packet in lst_dict_packet:
            # 判断是否是Packet
            if isinstance_dict(dict_packet) != OBJ_DctPkt:
                num -= 1
                continue
            temp_packet = Packet() 
            # dict-> Packet: dict多余的部分不会被赋予该对象,
            ## 如label, dataset_type, datatype
            temp_packet.fromDict(dict_packet) 
            self.lst_packets.append(temp_packet)
        # 输出从db:col中读取#个包
        if num == 0:
            self.print_wrong(f"{self.print_name}: create from [{mydb.database_name}: {mydb.collection_name}]: fail. no packets read.")
        else:
            self.print_right(f"{self.print_name}: create from [{mydb.database_name}: {mydb.collection_name}]: success! {num} packets read.")
        s = " "
        self.print_info(f"\tINFO: has {len(self.lst_packets)} packets in total now")

    def save_to_mongodb(self, mydb: MyMongoDB) -> None:
        db_name = mydb.database_name
        col_name = mydb.collection_name
        if len(self.lst_packets) == 0:
            self.print_wrong(f"{self.print_name}: save to [{db_name}: {col_name}]: fail. no packet to save")
            return
        lst_dict_packet: List[Dict] = []
        for packet in self.lst_packets:
            dict_packet = packet.toDict()
            lst_dict_packet.append(dict_packet)
        # if rewriteID:
        #     startID = 1
        #     for dict_packet in lst_dict_packet:
        #         dict_packet["_id"] = startID
        #         startID += 1
        mydb.insert(lst_dict_packet)
        self.print_right(f"{self.print_name}: save to [{db_name}: {col_name}]: success! {len(lst_dict_packet)} packets saved")
    
    def addread_from_mongodb(self, mydb: MyMongoDB, query: Dict = {}):
        lst_dict_packet = mydb.find(query)
        num = len(lst_dict_packet)
        for dict_packet in lst_dict_packet:
            # 判断是否是Packet
            if isinstance_dict(dict_packet) != OBJ_DctPkt:
                num -= 1
                continue
            temp_packet = Packet() 
            # dict-> Packet: dict多余的部分不会被赋予该对象,
            ## 如label, dataset_type, datatype
            temp_packet.fromDict(dict_packet) 
            self.lst_packets.append(temp_packet)
        # 输出从db:col中读取#个包
        if num == 0:
            self.print_wrong(f"{self.print_name}: addread from [{mydb.database_name}: {mydb.collection_name}]: fail. no packets read.")
        else:
            self.print_right(f"{self.print_name}: addread from [{mydb.database_name}: {mydb.collection_name}]: success! {num} packets read.")
        self.print_info(f"\tINFO: has {len(self.lst_packets)} packets in total after addread")
    
    def print_lst_packets(self, num = 10):
        self.print_right(f"{self.print_name}: print all pkts-------------")
        for packet in self.lst_packets[:num]:
            packet.print()
        self.print_info(f"\tINFO: has {len(self.lst_packets)} packets in total ")
        self.print_right(f"{self.print_name}: print end------------------")
    
    

if __name__ == "__main__":
    
    # 创建日志test, 保存在./log/test.log
    mylogger = MyLogger(name="test_log", log_filename="test_log.log")
    print("-----------\n")

    #创建packet 处理器
    PktMngr = PacketsManager(name="test_packet", logger=mylogger)
    # 创建packet
    p_1 = Packet()
    p_1.create(["A1_ip","B1_ip","A1_port","B1_port","1"], 100, 5, int(time.time()))
    p_2 = Packet()
    p_2.create(["A2_ip","B2_ip","A2_port","B2_port","-1"], 200, 10, int(time.time()))
    # 添加packet
    PktMngr.append(p_1)
    PktMngr.append(p_2)
    print("-----------\n")

    # 创建数据库
    mydb = MyMongoDB("test_packet", "test_black_packet")
    # 先删除数据库中之前的所有数据
    mydb.delete_all()
    mydb2 = MyMongoDB("test_packet", "test_black_packet")
    # 保存packet数据集到数据库
    PktMngr.save_to_mongodb(mydb)
    print("-----------\n")
    
    # 数据集逐个打印packet、更新packet、再打印packet
    for packet in tqdm(PktMngr.lst_packets):
        print("修改前：")
        packet.print()
        packet.update(110, 111, 1) # 此处修改，则 PktMngr.lst_packets 也会被修改
        print("修改后：")
        packet.print()
    print("-----------\n")

    # 从数据库读取packet 附加到packet数据集
    PktMngr.addread_from_mongodb(mydb=mydb,query={"fivetuple":["A1_ip","B1_ip","A1_port","B1_port","1"]})
    PktMngr.print_lst_packets()
    # 删除数据库
    mydb.delete_all()
    PktMngr.save_to_mongodb(mydb= mydb)
    
    mydb.print_front(10)