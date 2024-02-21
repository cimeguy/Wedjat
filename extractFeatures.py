# -*- encoding: utf-8 -*-
'''
@File    :   extractFeatures.py
@Time    :   2022/11/26 21:45:26
@Author  :   Hannah 
@Version :   1.0
@Contact :   1196033301@qq.com
@Desc    :   extract features from flows/ bags/ packets 
                for different models
'''

# here put the import lib

# here put the import lib

# sys import
import os, sys
from tqdm import tqdm
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import time
# here put the import lib
from ABCClass import Operation
from preProcess import *
from aggreate import *
from structure.Flow  import *
from structure.Packet import *
from structure.Bag import *

# utils/
from utils.MyColor import *
from utils.MyLogger import *

class PacketFeatures(Operation): 
    '''
    Operation PacketFeatures
    提取包粒度的特征
    --------------------------------
    Paramters:
    logger: MyLogger
    '''
    def __init__(self, logger: MyLogger = None) -> None:
        Operation.__init__(self,logger=logger,print_name="｜PacketFeatures｜")

    def _get_features_from_oneflow(self, flow: Flow)-> Tuple[List[List[float]],List[int]]:
        # return features , labels  for one flow
        # features: List[List[float]]
        lst_pktFeats_oneflow = []
        lst_pkts = flow.lst_packets
        for pkt_no, this_pkt in enumerate(lst_pkts[:]):
            if pkt_no == 0:
                first_pkt = lst_pkts[0]
                first_feats = [first_pkt.length,first_pkt.direction, 0.0]
                lst_pktFeats_oneflow.append(first_feats)
                continue
            pre_pkt = lst_pkts[pkt_no-1]
            ipt = this_pkt.timestamp - pre_pkt.timestamp
            this_feats = [this_pkt.length,this_pkt.direction, ipt]
            lst_pktFeats_oneflow.append(this_feats)
        # labels
        lst_pktLabels_oneflow = [flow.label] * len(flow.lst_packets)
        return lst_pktFeats_oneflow, lst_pktLabels_oneflow

    def get_features_from_flows(self, lst_flow: List[Flow], scalar=None)-> Tuple[List[List[float]],List[int]]:
        lst_pktFeats_allflows = []
        lst_pktLabels_allflows= []
        for flow in lst_flow:
            lst_pktFeats_oneflow, lst_pktLabels_oneflow = self._get_features_from_oneflow(flow)
            lst_pktFeats_allflows.extend(lst_pktFeats_oneflow)
            lst_pktLabels_allflows.extend(lst_pktLabels_oneflow)
        
        return lst_pktFeats_allflows, lst_pktLabels_allflows

    def get_features_from_onebag(self, lst_flow: List[Flow], scalar=None)-> Tuple[List[List[float]],List[int]]:
        # 输入：一个bag中的flows : bag.lst_flows
        lst_pktFeats_allflows = []
        lst_pktLabels_allflows= []
        for flow in lst_flow:
            lst_pktFeats_oneflow, lst_pktLabels_oneflow = self._get_features_from_oneflow(flow)
            lst_pktFeats_allflows.append(lst_pktFeats_oneflow)
            lst_pktLabels_allflows.append(lst_pktLabels_oneflow)
        
        return lst_pktFeats_allflows, lst_pktLabels_allflows

# class ExtractFeatures(Operation):

#     def __init__(self, logger: MyLogger = None) -> None:
#         Operation.__init__(self,logger=logger,print_name="｜ExtractFeats｜")
        
#     def getfeatures(data:List,model:str, logger: MyLogger)->Tuple[List,List]:
#         if model == "PacketFeatures":
#             extractor = PacketFeatures(logger=logger)
#             feats, labels = extractor.get_features_from_flows(data)
#         # return features, labels
#         return feats, labels

if __name__ == "__main__":
    log_filename = "test.log"
    dataset_dirname = "mytest"
    blackset_dirname = "test_black" # in dir[dataset_name]
    whiteset_dirname = "test_white" # in dir[dataset_name]
    blackset_dirpath = os.path.join(dataset_dir, dataset_dirname, blackset_dirname)
    whiteset_dirpath = os.path.join(dataset_dir, dataset_dirname, whiteset_dirname)
    # 创建日志记录器
    mylogger = MyLogger(log_filename = log_filename ,filemode="w")
    # 根据dataset和blackset连接数据库
    mydb = MyMongoDB(dataset_dirname, blackset_dirname)
    flowwithdb = FlowManager(logger=mylogger,name="flow&db")
    flowwithdb.create_from_mongodb(mydb=mydb)
    lst_flows = flowwithdb.get_datas()
    # white 
    white_files = ReadFile(logger=mylogger).get_allfile(whiteset_dirpath)
    print(lst_flows[0])
    # 提取包粒度的特征
    packetFeats = PacketFeatures(logger=mylogger)
    lst_pktFeats, lst_pktLabels = packetFeats.get_features_from_flows(lst_flows)
    np_pktFeats = np.array(lst_pktFeats) # (n_samples, n_features) = (4929, 3)
    print(np_pktFeats[:10])
    # scaler = preprocessing.StandardScaler().fit(np_pktFeats) 
    scaler = preprocessing.MinMaxScaler().fit(np_pktFeats) 
    np_pktFeats = scaler.transform(np_pktFeats)
    print("归一化")
    print(np_pktFeats[:10])
    # KMeans聚类
    model = KMeans(n_clusters=3)
    y_pred = model.fit_predict(np_pktFeats)

    # 画图显示样本数据
    plt.figure('Kmeans', facecolor='lightgray')
    plt.title('Kmeans', fontsize=16)
    plt.xlabel('X', fontsize=14)
    plt.ylabel('Y', fontsize=14)
    plt.tick_params(labelsize=10)
    plt.scatter(np_pktFeats[:, 0], np_pktFeats[:, 2], s=80, c=y_pred, cmap='brg', label='Samples')
    plt.legend()
    plt.show()

    thistime = time.time()
    figname = f"kmeans_T{thistime}.png"

    plt.savefig(os.path.join(figures_dir, figname))
    print(f"save fig done:{figname}")
