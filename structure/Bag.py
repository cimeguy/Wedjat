
from typing import *
import copy
import os,sys
from tqdm import tqdm

# 本级目录模块的import
this_filepath = __file__ # 此文件的路径: exp_Robust/utils/MyColor.py
this_dirpath = os.path.dirname(this_filepath) # 此文件所在文件夹/目录的路径: exp_Robust/utils
sys.path.append(this_dirpath) 
# e.g. import
from Packet import *
from DataTool import *
from Flow import *

# 上级目录模块的import: 将上级目录天加入环境变量，以便引入其他模块
parent_dirpath = os.path.join(this_dirpath, "..") # 加上..表示当前路径的上层目录:exp_Robust/
sys.path.append(parent_dirpath)  # 将上级目录添加到sys.path中，使得上级目录的模块可以被import
from config import *
from utils.MyColor import *
from utils.MyMongoDB import *


class Bag:
    def __init__(self) -> None:
        
        self.twotuple: List = [] # [src_ip, dst_ip]
        self.belongfile: str = "" # 所属文件名
        self.label = "" # 1 or 0 or UNKNOWN, fixed by human or first flow  
        self.type_dataset = ""  # train or test
        self.type_data = 'Bag' # fixed 
        self.lst_flows:List[Flow] = [] # flow更新 ，则相应的包也会更新
        self.desc = {}
    
    def create(self,firstflow: Flow) -> object:
        src_ip, dst_ip, belongfile= firstflow.get_BagID()
        self.twotuple = [src_ip, dst_ip]
        self.belongfile = belongfile
        self.label = firstflow.label
        self.lst_flows.append(firstflow)
        return self

    def get_BagID(self) -> Tuple:
        return (self.twotuple[0],self.twotuple[1],self.belongfile)
    def get_BagID_reversed(self) -> Tuple:
        return (self.twotuple[1],self.twotuple[0],self.belongfile)

    def append(self, flow: Flow) -> bool:
        # 添加flow到bag中
        if flow.get_BagID() != self.get_BagID() and flow.get_BagID_reversed() != self.get_BagID():
            return False
        self.lst_flows.append(flow)
        return True

    def finish(self) -> object:
        # 在添加完所有的flow后，对lst_flows进行排序
        # # 要求对each_flow的packet的顺序保证正确
        self.lst_flows.sort(key=lambda x: x.lst_packets[0].timestamp)
        return self
    
    def toDict(self)->Dict:
        dict_bag =copy.deepcopy(self.__dict__)# 必须深复制，否则会改变原来的值
        dict_bag['lst_flows'] = []
        if len(self.lst_flows)>0:
            for flow in self.lst_flows:
                dict_bag['lst_flows'].append(flow.toDict())
        return dict_bag

    def fromDict(self, dict_bag: Dict):
        self.__dict__ = copy.deepcopy(dict_bag) # 必须深复制，否则会改变原来的值
        # 创建lst_flows
        self.lst_flows = []
        for dict_flow in dict_bag['lst_flows']:  # 不能写列表生成式
            temp_flow = Flow()
            temp_flow.fromDict(dict_flow)
            self.lst_flows.append(temp_flow)
        # 对流按时间顺序排序
        self.finish()
        # 返回Bag
        return self


class PrintBag(PrintTool):
    def __init__(self, print_color: PrintColor = PrintColor()) -> object:
        # 设置输出颜色的模式
        PrintTool.__init__(self, print_color)
    
    def print_details(self, bag: Bag) -> None:
        # 打印bag的信息
        self.print("BagID: ", bag.get_BagID(), "label: ", bag.label, "type_dataset: ", bag.type_dataset)
        self.print("# of lst_flows: ", len(bag.lst_flows))
        seeflow = PrintFlow()
        for i, flow in enumerate(bag.lst_flows):
            print(f"f{i} in bag:")
            seeflow.print_abcinfo(flow)

    def print_view(self, bag: Bag)->None: # 看bag中所有packet的length, direction, timestamp
        self.print("BagID: ", bag.get_BagID(), "label: ", bag.label, "type_dataset: ", bag.type_dataset)
        self.print("# of lst_flows: ", len(bag.lst_flows))
        for i, flow in enumerate(bag.lst_flows):
            for j, packet in enumerate(flow.lst_packets):
                print(f"flow{i} packet{j}:{packet.get_data()}")
            print("\n")


class BagManager(DataManager):
    def __init__(self,lst_bag:List[Bag]=[], name: str="bag&db", logger: MyLogger=None) -> None:
        DataManager.__init__(self, logger, print_name=f"|BagMngr「{name}」|")
        self.lst_bags: List[Bag] = lst_bag
    
    def get_datas(self) -> List[Bag]:
        return self.lst_bags
        
    def create_from_lst_bag(self, lst_bag: List[Bag]):
        self.lst_bags = lst_bag
        return self

    def append(self, bag: Bag):
        # 添加bag到lst_bags中
        self.lst_bags.append(bag)
      
    def extend(self, lst_bag: List[Bag]):
        # 添加bag到lst_bags中
        self.lst_bags.extend(lst_bag)
    
    def clear(self):
        self.lst_bags = []

    def cnt(self):
        return len(self.lst_bags)
    
    def save_to_mongodb(self, mydb: MyMongoDB):
        # 保存到mongodb
        lst_dict_bag:List[Dict] = []
        for bag in self.lst_bags:
            dict_bag = bag.toDict()
            lst_dict_bag.append(dict_bag)
        try:
            mydb.insert(lst_dict_bag)
            self.print_right(f"{self.print_name}: save -> {mydb.print_name}: {len(lst_dict_bag)} bags saved.")

        except Exception as e:
            self.print_wrong("save bag wrong")
        self.print_info(f"\tINFO: {mydb.print_name} has {mydb.cnt_all()} datas in total")

    # 从数据库中读取数据
    def addread_from_mongodb(self, mydb: MyMongoDB, query: Dict = {}):
        # 如 query={"datatype":"black","dataset_type":"train"} 训练集的黑样本
        lst_dict_bag = mydb.find(query)
        num = len(lst_dict_bag)
        for dict_bag in lst_dict_bag:
            if isinstance_dict(dict_bag) != ObJ_DctBag:
                num -= 1
                continue
            bag = Bag()
            bag.fromDict(dict_bag)
            self.lst_bags.append(bag)
        self.print_right(f"{self.print_name}: addread <- {mydb.print_name}: {num} bags read.")
        self.print_info(f"\tINFO: has {self.cnt()} bags in total after addread")

    def create_from_mongodb(self, mydb: MyMongoDB, query: Dict = {}):
        self.lst_bags = []
        lst_dict_bag = mydb.find(query)
        num = len(lst_dict_bag)
        for dict_bag in lst_dict_bag:
            if isinstance_dict(dict_bag) != ObJ_DctBag:
                num -= 1
                continue
            bag = Bag()
            bag.fromDict(dict_bag)
            self.lst_bags.append(bag)
        self.print_right(f"{self.print_name}: create <- {mydb.print_name}: {self.cnt()} flows read.")
        return self
    

if __name__ == "__main__":
    bag1 = Bag()
    packet1 = Packet()
    packet2 = Packet()
    packet1.create(fivetuple=['Ai','Bi','Ap',"Bp","10"], length=100, timestamp=1.1, direction=1)
    packet2.create(fivetuple=['Ai','Bi','Ap',"Bp","10"], length=200, timestamp=1.2, direction=1)
    flow1 = Flow()
    flow1.create(packet1,belongfile="file1")
    flow1.append(packet2)
    PrintFlow().print_details(flow1)

    bag1.create(flow1)
    bag1.append(flow1)
    bag1.append(flow1)

    PrintBag().print_details(bag1)
    bagMgr = BagManager()
    bagMgr.append(bag1)
    bagMgr.save_to_mongodb(MyMongoDB("test_bag","test_bag_black"))
    bagMgr.addread_from_mongodb(MyMongoDB("test_bag","test_bag_black"))
    for bag in bagMgr.get_datas():
        print("========")
        PrintBag().print_view(bag)