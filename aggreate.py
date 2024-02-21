# -*- encoding: utf-8 -*-
'''
@File    :   aggreate.py
@Time    :   2022/11/10 02:32:03
@Author  :   Hannah 
@Version :   1.0
@Contact :   1196033301@qq.com
@Desc    :   Aggreator: Packet <- Flow <-> Bag (switch grain)
             Filter: filter on 3 grain based rules
'''

# sys import
# 系统导入
import os, sys
from tqdm import tqdm
# here put the import lib
# 本级目录模块
from preProcess import *
from ABCClass import Operation
# 下级目录模块
# structure/
from structure.Flow  import *
from structure.Packet import *
from structure.Bag import *
# utils/
from utils.MyColor import *
from utils.MyLogger import *


class Filter(Operation):
    '''
    Operation Filter
    --------------------------------
    Paramters:
    logger: MyLogger
    '''
    def __init__(self, logger: MyLogger = None) -> None:
        Operation.__init__(self,logger=logger,print_name="｜Filter「on_data」｜")

    def filter_on_lst_flows(self, lst_flow: List[Flow],rules={}) -> List[Flow]:
        '''
        TODO: filter on flow
        '''
        return lst_flow
        

class Aggreator(Operation):
    '''
    Operation aggregator
    --------------------------------
    Paramters:
    logger: MyLogger
    '''
    
    def __init__(self, logger: MyLogger=None) -> None:
        Operation.__init__(self,logger=logger,print_name="｜Aggreator「3grain」｜")  
        
    def aggregate(self,lst_flow:List[Flow])->List[Bag]:
        '''
        aggregate flow to bag
        --------------------------------
        Paramters:
            lst_flow: List[Flow]
        Return:
            List[Bag]
        '''
        dict_allbag :Dict[tuple, Bag]= {}
        for flow in tqdm(lst_flow):
            bagID: tuple = flow.get_BagID()
            bagID_rvsd: tuple = flow.get_BagID_reversed()
            if bagID in dict_allbag.keys() :
                dict_allbag[bagID].append(flow)
            elif bagID_rvsd in dict_allbag.keys():
                dict_allbag[bagID_rvsd].append(flow)
            else: # new bag
                bag = Bag()
                bag.create(flow)
                dict_allbag[bagID] = bag
        self.print_right(f"{self.print_name}: aggregate flow to bag successfully!")
        self.print_info(f"\tINFO: there are {len(dict_allbag.keys())} bags from {len(lst_flow)} flows.")
        return list(dict_allbag.values())

    def get_bags_from_flows(self, lst_flow:List[Flow]) -> List[Bag]:
        return self.aggregate(lst_flow)
        

    def get_flows_from_bags(self, lst_bag:List[Bag]) -> List[Flow]:
        lst_allflows = []
        for bag in lst_bag:
            lst_allflows.extend(bag.lst_flows)
        return lst_allflows

    def get_packets_from_flows(self, lst_flow:List[Flow]) -> List[Packet]:
        lst_allpackets = []
        for flow in lst_flow:
            lst_allpackets.extend(flow.lst_packets)
        return lst_allpackets

    def get_packets_from_bags(self, lst_bag:List[Bag]) -> List[Packet]:
        
        lst_allflows = self.get_flows_from_bags(lst_bag)
        return self.get_packets_from_flows(lst_allflows)
    
if __name__ == "__main__":
    log_filename = "test.log"
    dataset_dirname = "mytest"
    blackset_dirname = "test_black" # in dir[dataset_name]
    blackset_dirpath = os.path.join(dataset_dir, dataset_dirname, blackset_dirname)
    # 创建日志记录器
    mylogger = MyLogger(log_filename = log_filename ,filemode="w")

    # 根据dataset和blackset连接数据库，获得flows lst
    mydb = MyMongoDB(dataset_dirname, blackset_dirname)
    flowwithdb = FlowManager(logger=mylogger,name="flow&db")
    flowwithdb.create_from_mongodb(mydb=mydb)
    lst_flows = flowwithdb.get_datas()#flow的lst

    # 将flows聚合成bags
    switch_grain = Aggreator(logger=mylogger) # 聚合器
    lst_bags = switch_grain.aggregate(lst_flows)#  flow的lst -> bag的lst
    seebag = PrintBag()
    seebag.print_details(lst_bags[0])
    lst_packets = switch_grain.get_packets_from_bags(lst_bags)
    lst_packets[5].print()



