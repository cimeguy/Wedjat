# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2022/11/27 00:23:58
@Author  :   Hannah 
@Version :   1.0
@Contact :   1196033301@qq.com
@Desc    :   main
'''

# here put the import lib

# sys import
import os, sys
import json
from tqdm import tqdm
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
# here put the import lib
from config import *
from preProcess import *
from aggreate import *
from extractFeatures import *
from kernelkmeans import *


# structure/
from structure.Flow  import *
from structure.Packet import *
from structure.Bag import *
import scienceplots
import umap
import scienceplots # 如果不存在，需要pip install scienceplots
plt.style.available # 查看可用的样式    
plt.style.use(['science','ieee']) # 选择一个样式

# model
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.inference import BeliefPropagation, VariableElimination
import pandas as pd

# utils/
from utils.MyColor import *
from utils.MyLogger import *
from utils.MyMongoDB import *
import random
log_filename = "test.log"
# 使用的数据集名
root_dataset_dirpath = "/data/users/gaoli/exp_Robust/datasets"
dataset_dirname = "mytest" # "mytest"
blackset_dirname = "test_black" # in dir[dataset_name]
whiteset_dirname = "test_white" # in dir[dataset_name]
blackset_dirpath = os.path.join(dataset_dir, dataset_dirname, blackset_dirname)
whiteset_dirpath = os.path.join(dataset_dir, dataset_dirname, whiteset_dirname)
RANDOMSEED = 2024
random.seed(RANDOMSEED) 

result_files = "/data/users/gaoli/exp_Robust/results/test.txt"
with open(result_files,"w") as f:
    # 生成当前的时间
    now_time = datetime.datetime.now()
    f.write(f"{now_time}")
c_0 = "#8dabdc"# 天蓝色
c_1 = "#c45c0b" # 桔红色
c_2 = "#e5b051" # 黄色
c_3 ="#364b61" # 深蓝色
c_4 = "#7671ac" # 深紫色
c_5 = "#da4d5c" # 粉红色
c_6 =  "#649993" # 蓝绿色
c_7 = "#bd9725" # 棕黄色
c_list = [c_0, c_1, c_2, c_3, c_4, c_5,c_6, c_7] 
# mycmap = ListedColormap(["#c82423", "#0088C4",
                        #   "#00A7D0", "#00C3C4", "#57784B", "#8AAD7C",
                        #   "#5DA37E","#5DA37E","#A5538B"])
mycmap = ListedColormap(c_list)
alpha = 0.8

Missing_packet = -1

def construct_base_edges(n, m) -> List[tuple]:
    # n*m ， n flows， each flow has m packets
    edges = []
    for i in range(1,n+1):
        for j in range(1,m+1):
            if i == 1 and j == 1:
                continue
            elif i == 1 and j!= 1:
                edges.append((f"Packet_{i}_{j-1}", f"Packet_{i}_{j}"))
            elif j == 1 and i!=1:
                edges.append((f"Packet_{i-1}_{j}", f"Packet_{i}_{j}"))
            else:
                edges.append((f"Packet_{i-1}_{j}", f"Packet_{i}_{j}"))
                edges.append((f"Packet_{i}_{j-1}", f"Packet_{i}_{j}"))
        
    return edges

def create_columns(n, m) -> List[str]:
    columns = []
    for i in range(1,n+1):
        for j in range(1,m+1):
            columns.append(f"Packet_{i}_{j}")
    return columns

class Model():

    def __init__(self, sample_flows_num=1, sample_packets_num=15):
        self.sample_flows_num = sample_flows_num
        self.sample_packets_num = sample_packets_num
        self.base_edges = construct_base_edges(self.sample_flows_num, self.sample_packets_num) # 基本图结构
        self.columns = create_columns(self.sample_flows_num, self.sample_packets_num)
        self.model =None #BayesianNetwork(self.base_edges)
        self.this_edges = None # 优化后的图结构
        # 创建一个空的 DataFrame 作为学习数据
        self.learning_dataframe = pd.DataFrame(columns=self.columns)
        
    
    def train(self, trainsamples: List[np.ndarray],estimator=MaximumLikelihoodEstimator,train_num=400):
        #### 为每个节点添加数据
        # 创建一个空的 DataFrame 作为学习数据
        self.learning_dataframe = pd.DataFrame(columns=self.columns)
       
        for index, onesample in enumerate(trainsamples[:train_num]):
            row_data = [] # 每个样本创建一行数据, onesample是一个样本，一个矩阵
            for each_flow in onesample:
                row_data.extend(each_flow)
            # print(row_data)
            # print(len(row_data))
            # 检查长度
            if len(row_data)>self.sample_flows_num*self.sample_packets_num:
                row_data = row_data[:self.sample_flows_num*self.sample_packets_num]
            # 添加样本数据
            self.learning_dataframe.loc[index] = row_data
            
        print("训练数据:",self.learning_dataframe.head())
        print("训练数据个数:",self.learning_dataframe.shape)
        # 直接参数学习
        # self.model.fit(self.learning_dataframe, estimator) # 最大似然学习参数
        #  BayesianNetwork(self.base_edges).fit(data, estimator=MaximumLikelihoodEstimator)
        # 优化图结构
        hc = HillClimbSearch(self.learning_dataframe)# hillclimbsearch
        best_model = hc.estimate(scoring_method=BicScore(self.learning_dataframe),white_list=self.base_edges) # 使用BIC评分 ,是否使用白名单white_list=self.base_edges
        # BDeuScore,K2Score
        # ExhaustiveSearch(data, scoring_method=bic)
        edges = best_model.edges() # 学到最优边
        print("base edges", self.base_edges)
        print("edges from hillclimbing:", edges)
        # input("pause")
        self.this_edges = edges # edges
        # edges -> model
        self.model = BayesianNetwork(self.this_edges)
        self.model.fit(self.learning_dataframe, estimator=MaximumLikelihoodEstimator)
        # print(self.model.nodes())
        self.model.check_model()

    def predict(self, onesample: List[np.ndarray]): # 输入一个样本，输出预测的结果
        # onesample是一个样本，一个矩阵,每行代表flow，每列代表packet
        row_data = []
        Flow_wise_right = []
        for each_flow in onesample:
            row_data.extend(each_flow)
        # 检查长度
        if len(row_data)>self.sample_flows_num*self.sample_packets_num:
            row_data = row_data[:self.sample_flows_num*self.sample_packets_num]
        
        # 使用 BeliefPropagation 进行推断
        try:
            inference_bp = BeliefPropagation(self.model) # 推断器
        # joint_inference = VariableElimination(self.model)
        except:
            print("BeliefPropagation Erroring, There is error in Data")
            return None, None
        # 设置evidence
        nodes = self.model.nodes()
        nodes = sorted(nodes, key=lambda x: (int(x.split('_')[1]), int(x.split('_')[2])))
        # print(nodes)
        firstNode_i = int(nodes[0].split("_")[1])
        firstNode_j = int(nodes[0].split("_")[2])

        evidence = {f'Packet_{firstNode_i}_{firstNode_j}': onesample[firstNode_i-1][firstNode_j-1]}
        cnt_predict_right = 0
        cnt_predict_wrong = 0
        infertime = []
        # all_posterior_prob = []
        # 推断
        flow_end = False
        for idx, node in enumerate(nodes[:]): #开始预测每个节点
            if idx == 0:
                continue
            i = int(node.split("_")[1])
            j = int(node.split("_")[2])
            nodevalue = onesample[i-1][j-1]
            
            # print(f"{node}'s real value:",nodevalue)
            
            # flow-wise 
            pre_node = nodes[idx-1]
            pre_i = int(pre_node.split("_")[1])
            if i > pre_i:# step into next flow
                with open(result_files,"a") as f:
                    str_2_file = f"{cnt_predict_right/(cnt_predict_right+cnt_predict_wrong)},{cnt_predict_right}/{cnt_predict_right+cnt_predict_wrong}"
                    f.write(str_2_file+"\n")
                print(f"flow_{pre_i} right ratio:",cnt_predict_right/(cnt_predict_right+cnt_predict_wrong),f"\t{cnt_predict_right}/{cnt_predict_right+cnt_predict_wrong}")
                Flow_wise_right.append(cnt_predict_right/(cnt_predict_right+cnt_predict_wrong))
            #     print("right ratio:",cnt_predict_right/(cnt_predict_right+cnt_predict_wrong))
                if nodevalue != Missing_packet: # 表示这个包不是缺失值，也就是说进入下一条流了
                    cnt_predict_right = 0  
                    cnt_predict_wrong = 0
                else: # 当前流的第一个节点就是missing，代表整个bag结束
                    flow_end = True
                    cnt_predict_right = 0  
                    cnt_predict_wrong = 0
                    break

            if nodevalue != Missing_packet: # 不是缺失值
                starttime = time.time()
                try:
                    query_bp = inference_bp.map_query(variables=[node], evidence=evidence)
                except:
                    # print("KeyError")
                    cnt_predict_wrong += 1 # 无法预测 ，意味着没有这个值
                    continue
                lasttime = time.time()-starttime
                infertime.append(lasttime)
                # print("\t  prediction value:", query_bp[node])
                if query_bp[node] == nodevalue:
            
                    cnt_predict_right += 1
                else:
                    cnt_predict_wrong += 1
            # 更新evidence
                evidence[node] = nodevalue
        
        
        if flow_end == False: # 最后一个流的统计结果
            Flow_wise_right.append(cnt_predict_right/(cnt_predict_right+cnt_predict_wrong))
            print("flow_end right ratio:",cnt_predict_right/(cnt_predict_right+cnt_predict_wrong),f"\t{cnt_predict_right}/{cnt_predict_right+cnt_predict_wrong}")
       
            with open(result_files,"a") as f:
                str_2_file = f"{cnt_predict_right/(cnt_predict_right+cnt_predict_wrong)},{cnt_predict_right}/{cnt_predict_right+cnt_predict_wrong}"
                f.write(str_2_file+"\n")
        return Flow_wise_right, infertime
        
    # print(df)

class Experiment(Operation):

    def __init__(self, logger: MyLogger = None, print_name: str = "｜Experiment「test」｜") -> None:
        Operation.__init__(self,logger,print_name)
        self.print_attention(f"实验 {print_name} 开始...")
    
    def _get_subset_dirpath(self, this_dataset_dirname:str, this_subset_dirname:str, root_dataset_dirpath:str = dataset_dir) -> str:
        '''
        get the dirpath of the subset
        path: dataset_dir/dataset_name/subset_name
        '''
        subset_dirpath = os.path.join(root_dataset_dirpath,this_dataset_dirname, this_subset_dirname)
        return subset_dirpath
    

    def load_data(self, this_dataset_dirname:str, this_subset_dirname:str,root_dataset_dirpath:str = dataset_dir, onlyfromfile=False) -> List:
        
        '''
        load data from the subset
        path: dataset_dir/dataset_name/subset_name
        '''
        print("-----onlyfromfile:", onlyfromfile)
        # 不止从文件中读取
        # # 数据库中存在， 从数据库中读取
        # # 否则， 从文件中读取
        if isexists_db_col(this_dataset_dirname, this_subset_dirname) and onlyfromfile==False:
            self.print_right(f"数据库「{this_dataset_dirname}:{this_subset_dirname}」已存在，连接数据库")
            mydb = MyMongoDB(database=this_dataset_dirname, collection=this_subset_dirname)
            flow_read_from_db= FlowManager(logger=mylogger,name="db->flow")
            flow_read_from_db.create_from_mongodb(mydb)
            lst_flows = flow_read_from_db.get_datas()
            self.print_info(f"\t来自数据库「{this_dataset_dirname}/{this_subset_dirname}」的样本流数量:{len(lst_flows)}")
            return lst_flows
        # 只从文件夹中读取，并询问是否覆盖数据库
        elif onlyfromfile and isexists_db_col(this_dataset_dirname, this_subset_dirname):
            cover_db_col = input("数据库已存在，但从文件读取数据，是否覆盖数据库？(y/n)")
            if cover_db_col == "y":
                self.print_attention(f"覆盖数据库「{this_dataset_dirname}:{this_subset_dirname}」")
                mydb = MyMongoDB(database=this_dataset_dirname, collection=this_subset_dirname)
                mydb.delete_all()
            else:
                self.print_attention(f"不覆盖数据库「{this_dataset_dirname}:{this_subset_dirname}」,从文件读取的数据会加入数据库中")
                

        else: # 不存在数据库，创建数据库
            self.print_right(f"数据库「{this_dataset_dirname}:{this_subset_dirname}」不存在，创建数据库")
        # elif/else -> 从文件夹中读取
            
        this_subset_dirpath = self._get_subset_dirpath(this_dataset_dirname, this_subset_dirname, root_dataset_dirpath)
        self.print_right(f"读取文件夹:{this_subset_dirpath}")
        # 读取文件夹
        # # 创建文件夹处理器： 得到文件夹下的所有文件的路径
        lst_files = ReadFile(logger=mylogger).get_allfile(dirpath=this_subset_dirpath)
        # # 创建流处理器： 读取文件，创建流
        lst_flows = Processor(logger=mylogger, auto_label=True).process_allfile(lst_files)
        # # 创建流管理器： 存储流 
        flow_store_to_db = FlowManager(lst_flows,logger=mylogger,name="flow->db")
        self.print_right(f"文件->流读取完成，存入数据库")
        # # 链接数据库
        mydb = MyMongoDB(database=this_dataset_dirname, collection=this_subset_dirname)
        # # 流管理器存储流到数据库
        flow_store_to_db.save_to_mongodb(mydb)
        self.print_right(f"流->数据库存储完成")
        self.print_info(f"来自文件的样本流数量:{len(lst_flows)}")
        self.print(f"数据库「{this_dataset_dirname}:{this_subset_dirname}」的样本数量:{len(lst_flows)}")
        return lst_flows
        
def get_purity(ytrue:List, ypred:List)->List:
    '''
    get purity
    '''
    # get the number of clusters
    n_clusters = len(set(ypred))
    # get the number of samples
    n_samples = len(ytrue)
    blacknum_clusters = [0] * n_clusters # 第i个簇中黑样本的数量
    whitenum_clusters = [0] * n_clusters
    for i in range(n_samples):
        if ytrue[i] == BLACK:
            blacknum_clusters[ypred[i]] += 1 # 第ypred[i]个簇中黑样本的数量, i是样本的索引
        else:
            whitenum_clusters[ypred[i]] += 1
    # get the purity of each cluster
    purity_clusters_b = [0] * n_clusters
    purity_clusters_w = [0] * n_clusters
    purity_clusters_maxeach = [0] * n_clusters
    label_maxpurity_maxeach = ['b or w'] * n_clusters
    for i in range(n_clusters):
        purity_clusters_b[i] = blacknum_clusters[i] / (blacknum_clusters[i] + whitenum_clusters[i])
        purity_clusters_w[i] = whitenum_clusters[i] / (blacknum_clusters[i] + whitenum_clusters[i])
        purity_clusters_maxeach[i] = max(purity_clusters_b[i], purity_clusters_w[i])
        if max(purity_clusters_b[i], purity_clusters_w[i]) == purity_clusters_b[i]:
            label_maxpurity_maxeach[i] = BLACK
        else:
            label_maxpurity_maxeach[i] = WHITE
    # get the avg_purity of the whole clusters
    purity_clusters_all_avg = sum(purity_clusters_maxeach) / n_clusters
    cnt_each_cluster = [0] * n_clusters
    for i in ypred:
        cnt_each_cluster[i] += 1
    print("cnt_each_cluster:", cnt_each_cluster)
    return purity_clusters_b, purity_clusters_w, purity_clusters_all_avg, label_maxpurity_maxeach
    
def get_metric_PktClutering(ytrue:List, ypred:List)->float:
    # p: purity
    # c: cluster
    p_c_b, p_c_w, p_c_avgmax, label_maxc = get_purity(ytrue, ypred)
    if sum(label_maxc) > 0 and sum(label_maxc) < 1*len(label_maxc):
        return p_c_avgmax
    else:
        return 0.

def get_mapclass_from_ypred(ytrue: List, ypred: List, threshold_black: float = 0.6,threshold_white: float = 0.6) -> List:
    p_c_b, p_c_w, p_c_avgmax, label_maxc = get_purity(ytrue, ypred)
    y_mapclass = [0] * len(ytrue)
    num_black = 0
    num_white = 0
    num_unknow = 0
    for idx, c in enumerate(ypred):
        if p_c_b[c] > threshold_black:
            num_black += 1
            y_mapclass[idx] = BLACK
        elif p_c_w[c] > threshold_white:
            num_white += 1
            y_mapclass[idx] = WHITE
        else:
            y_mapclass[idx] = 2
            num_unknow += 1
    print(f"MapClass:\tblack:{num_black},white{num_white},unknown:{num_unknow}")
    return y_mapclass




if __name__ == "__main__":
    
    #提前启动数据库：
    # 命令行输入：mongod --dbpath ~/mongodb_store/lib/ --logpath ~/mongodb_store/log/mongodb.log --fork

    # 创建日志记录器
    mylogger = MyLogger(log_filename = log_filename ,filemode="w")
    
    # 创建实验
    exp = Experiment(logger=mylogger)
    onlyfromfile = False  # flag:是否只从文件读取数据
    
    print("onlyfromfile:", onlyfromfile)
    lst_black_flows = exp.load_data(dataset_dirname, blackset_dirname,root_dataset_dirpath, onlyfromfile)
    lst_white_flows = exp.load_data(dataset_dirname, whiteset_dirname,root_dataset_dirpath, onlyfromfile)
    print(lst_black_flows[0].get_BagID())
    # sys.exit(0)
    exp.print_info(f"黑样本流数量:{len(lst_black_flows)}")
    exp.print_info(f"白样本流数量:{len(lst_white_flows)}")
    # 提取包粒度的特征
    packetFeats = PacketFeatures(logger=mylogger)
    lst_pktFeats_black, lst_pktLabels_black = packetFeats.get_features_from_flows(lst_black_flows)
    exp.print_info(f"黑样本包数量：{len(lst_pktFeats_black)}")
    lst_pktFeats_white, lst_pktLabels_white = packetFeats.get_features_from_flows(lst_white_flows)
    exp.print_info(f"白样本包数量：{len(lst_pktFeats_white)}")
    
    # # 数据包大小的分布情况#####
    # packet_size_pdf = []
    # for i in range(len(lst_pktFeats_black[:2000])):
    #     packet_size_pdf.append(lst_pktFeats_black[i][0] * lst_pktFeats_black[i][1])
    # for i in range(len(lst_pktFeats_white[:2000])):
    #     packet_size_pdf.append(lst_pktFeats_white[i][0] * lst_pktFeats_white[i][1])
    # with open("/data/users/gaoli/exp_Robust/datasets/json/packet_size_pdf_private.json","w")as f:
    #     json.dump(packet_size_pdf,f)
    # sys.exit(0)
    # ########################

    # 取固定个数个packet
    num_pkt_each =int(input("聚类的包个数(推荐4000):")) # 4000# 黑白packet各取num_pkt_each个packet
    np_pktFeats_black = np.array(lst_pktFeats_black)[:num_pkt_each,:]
    np_pktFeats_white = np.array(lst_pktFeats_white)[:num_pkt_each,:]
    np_pktLabels_black = np.array(lst_pktLabels_black)[:num_pkt_each]
    np_pktLabels_white = np.array(lst_pktLabels_white)[:num_pkt_each]
    # 黑白packet拼接为所有packet：沿着第 0 轴拼接，即行数增加
    np_pktFeats_all = np.concatenate((np_pktFeats_black, np_pktFeats_white), axis=0) # output: 8000 * 3
    # 8000 * 3
    # 标签拼接 y_true   
    np_pktLabels_all = np.concatenate((np.array(np_pktLabels_black), np.array(np_pktLabels_white)), axis=0)
    y_true = np_pktLabels_all
    # 特征归一化
    scaler = preprocessing.MinMaxScaler()
    np_pktFeats_all = scaler.fit_transform(np_pktFeats_all)  
    print(np_pktFeats_all.shape)
    
    model_clustering = KMeans(n_clusters=7,random_state=RANDOMSEED+1)
    y_pred = model_clustering.fit_predict(np_pktFeats_all)
    purity_each_cluster_b, purity_each_cluster_w, avg_purity_max_cluster, label_each_cluster = get_purity(np_pktLabels_all, y_pred)
    exp.print_info(f"每个簇的黑样本纯度:{purity_each_cluster_b}")
    exp.print_info(f"每个簇的白样本纯度:{purity_each_cluster_w}")
    exp.print_info(f"每个簇的最大纯度的平均值:{avg_purity_max_cluster}")
    exp.print_info(f"每个簇的最大纯度的标签:{label_each_cluster}")
    # 所有样本到簇心的距离
    all_distances_trainset = model_clustering.transform(np_pktFeats_all) # output: 8000 * 7
    
    # 计算指标并输出
    metric = get_metric_PktClutering(np_pktLabels_all, y_pred)
    exp.print_attention(f"聚类的指标:{metric}")
    y_mapclass = get_mapclass_from_ypred(y_true, y_pred,threshold_black=0.51, threshold_white=0.56)
    
    # 聚合
    switch_grain = Aggreator(logger=mylogger) # 聚合器
    lst_black_bags = switch_grain.aggregate(lst_black_flows)#  flow的lst -> bag的lst
    print("================black================")
    lst_white_bags = switch_grain.aggregate(lst_white_flows)
    print("================white================")
    # 处理bag到packet的矩阵的映射
    MAX_FLOWS = 3# 3
    MAX_PACKETS = 10 # 15

    VALID_FLOW_PACKETS = 8 # 8, 有效流的最小包数

    offset = 5 # 5, 往右偏移offset个包， offset, offset+MAX_PACKETS


    # 构造训练数据
    allsamples = [] # 所有训练数据
    for onebag in tqdm(lst_white_bags[:], desc="处理白样本数据"):

        feats_flows, label = packetFeats.get_features_from_onebag(onebag.lst_flows)
        # fe的每个元素是流，流的每个元素是包，包的元素是特征
        discarded_thisbag = False
        onesample = []
        for each_flow in feats_flows:
            # 将each flow转化为矩阵
            np_each_flow = np.array(each_flow) # 一个流的矩阵样本 n_packets * 3
            if np_each_flow.shape[0] <= VALID_FLOW_PACKETS: # 不符合有效流过滤
                discarded_thisbag = True
                continue

            # print(np_each_flow.shape)
            # print(np_each_flow)
            scaler_np_each_flow = scaler.transform(np_each_flow) # 归一化
            # print(scaler_np_each_flow.shape)
            # print(scaler_np_each_flow)
            # 聚类
            output = list(model_clustering.predict(scaler_np_each_flow))
            # model.predict(packet)
            # output 不足补-1
            
            if len(output) < MAX_PACKETS+offset:
                output.extend([Missing_packet] * (MAX_PACKETS+offset - len(output)))
            # output 超过截断
            if len(output) > MAX_PACKETS+offset:
                output = output[:MAX_PACKETS+offset]
            output = output[offset:]
            # print(len(output))
            onesample.append(output)
        # onesample 不足补-1
        if onesample == [] or discarded_thisbag:
            continue
        if len(onesample) < MAX_FLOWS:
            onesample.extend([[Missing_packet] * MAX_PACKETS] * (MAX_FLOWS - len(onesample)))
        if len(onesample) > MAX_FLOWS:
            onesample = onesample[:MAX_FLOWS]
        onesample = np.array(onesample) # 一个bag的矩阵样本
    
        allsamples.append(onesample)

    print("allsamples:", len(allsamples))
    print("each bag:", len(allsamples[0]))
    print("each flow:", len(allsamples[0][0]))
    
    # 建bayesnet
    ## ##################   训练 # ################## 

    # allsamples = random.sample(allsamples, len(allsamples)) # 打乱
    model = Model(sample_flows_num=MAX_FLOWS, sample_packets_num=MAX_PACKETS)
    print("all white:", len(allsamples))
    train_num = int(input("# for train:（1000）")) 
    
    model.train(allsamples[:train_num],train_num=train_num)
    print(model.model.check_model())
    # 在训练集上推断以确定阈值
   
    lst_right_packets_ratio_traindata = []
    print("_______training output ______")
    for onesample in tqdm(allsamples[:train_num],f"{MAX_FLOWS}x{MAX_PACKETS}|设置训练数据阈值"):
        flowwise_right, infertimes = model.predict(onesample)
        # 每条流的预测正确的包的比例
        if flowwise_right == None:
            continue
        lst_right_packets_ratio_traindata.extend(flowwise_right)
        # 注意排除-1的数据包 也就是缺失的情况
        # [0.5263157894736842, 0.7894736842105263, 0.7368421052631579, 0.5263157894736842, 0.47368421052631576]
    print("推断时间:",infertimes)
    print("平均推断时间：", sum(infertimes)/len(infertimes))
    print("总推断时间：", sum(infertimes))
    with open("/data/users/gaoli/exp_Robust/results/white_tr.txt","w") as f:
        for i in lst_right_packets_ratio_traindata:
            f.write(str(i)+",")
    # model.predict(allsamples[0])
    # sys.exit(0)
    print(lst_right_packets_ratio_traindata)
    T2_list = []
    for i in [5,10,20,30,40,50,60,70,80,90]:
        T2_list.append(np.percentile(lst_right_packets_ratio_traindata, i))
    
    threshold_black = sum(lst_right_packets_ratio_traindata) / len(lst_right_packets_ratio_traindata)
    input("pause:  训练阶段白样本阈值")
    print(T2_list)

     # [0.4513986013986014, 0.46373626373626375, 0.5242467718794835, 0.5392857142857143, 0.5625, 0.5929411764705882, 0.6666666666666666, 0.6666666666666666, 0.7457013574660634]
    ##################### 测试 ###################
    print("_______testing white______")
    lst_right_packets_ratio_testdata_white = []
    
    for onesample in tqdm(allsamples[train_num+1:],"测试白样本"):
        flowwise_right, infertimes = model.predict(onesample)
        # 每条流的预测正确的包的比例
        if flowwise_right == None:
            continue
        lst_right_packets_ratio_testdata_white.extend(flowwise_right)
        # 注意排除-1的数据包 也就是缺失的情况
        # [0.5263157894736842, 0.7894736842105263, 0.7368421052631579, 0.5263157894736842, 0.47368421052631576]
    print("推断时间:",infertimes)
    print("平均推断时间：", sum(infertimes)/len(infertimes))
    print("总推断时间：", sum(infertimes))
    print(lst_right_packets_ratio_testdata_white)
    with open("/data/users/gaoli/exp_Robust/results/white_te.txt","w") as f:
        for i in lst_right_packets_ratio_testdata_white:
            f.write(str(i)+",")
    
    #构造测试数据
    blacksamples = [] # 所有黑样本测试数据
    for onebag in tqdm(lst_black_bags[:],"处理测试数据->样本"):
        feats_flows, label = packetFeats.get_features_from_onebag(onebag.lst_flows)
        # fe的每个元素是流，流的每个元素是包，包的元素是特征
        discarded_thisbag = False   
        onesample = []
        for each_flow in feats_flows:
            # 将each flow转化为矩阵
            np_each_flow = np.array(each_flow)
            if np_each_flow.shape[0] <= VALID_FLOW_PACKETS: # 不符合有效流过滤
                discarded_thisbag = True
                continue
            scaler_np_each_flow = scaler.transform(np_each_flow)
            # 聚类
            output = list(model_clustering.predict(scaler_np_each_flow))
            # output 不足补-1
            if len(output) < MAX_PACKETS+offset:
                output.extend([Missing_packet] * (MAX_PACKETS+offset - len(output)))
            # output 超过截断
            if len(output) > MAX_PACKETS+offset:
                output = output[:MAX_PACKETS+offset]
            output = output[offset:]
            # print(len(output))
            onesample.append(output)
        # onesample 不足补-1
        if onesample == [] or discarded_thisbag:# 抛弃不符合条件的样本
            continue
        if len(onesample) < MAX_FLOWS:
            onesample.extend([[Missing_packet] * MAX_PACKETS] * (MAX_FLOWS - len(onesample)))
        if len(onesample) > MAX_FLOWS:
            onesample = onesample[:MAX_FLOWS]

        onesample = np.array(onesample) # 一个bag的矩阵样本
        blacksamples.append(onesample)    
        
    
       
    print("allsamples:", len(blacksamples))
    print("each bag:", len(blacksamples[7]))
    print("each flow:", len(blacksamples[5][0]))

    # ################## 测试黑 # ################## 
        # 建bayesnet
    exp.print_attention("黑样本")
    lst_right_packets_ratio_testdata = []
    for onesample in tqdm(blacksamples[:],"testing phase"):
        print(onesample)
        flowwise_right, infertime = model.predict(onesample)
        if flowwise_right == None:
            continue
        # 每条流的预测正确的包的比例
        lst_right_packets_ratio_testdata.extend(flowwise_right)
    # model.predict(allsamples[0])
    print(lst_right_packets_ratio_testdata)
    
    T2_black = []
    for i in [10,20,30,40,50,60,70,80,90,95,98]:
        T2_black.append(np.percentile(lst_right_packets_ratio_testdata, i))
    print("training_black_threshold",T2_black)
    with open("/data/users/gaoli/exp_Robust/results/black_all.txt","w") as f:
        for i in lst_right_packets_ratio_testdata:
            f.write(str(i)+",")

    print("Hyper-Parameters:",MAX_FLOWS,MAX_PACKETS,VALID_FLOW_PACKETS,offset)

    # evasion
    evading_flow = np.array([[1,1,2,1,1 ,1, 1, 2, 0, 0, 1, 0, 1, 4, 0],
                    [1, 2, 1, 2, 1, 0, 1, 1, 2, 2, 1, 1 ,1, 0 ,0],
                    [1, 2, 0 ,1, 2 ,1, 2, 0, 0, 0 ,1, 4 ,0 ,6 ,0]])
    flowwise_right, infertime = model.predict(evading_flow)