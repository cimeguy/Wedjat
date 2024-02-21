# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2022/11/27 00:23:58
@Author  :   Hannah 
@Version :   1.0
@Contact :   1196033301@qq.com
@Desc    :   this is for choosing the best clustering model
'''

# here put the import lib

# sys import
import os, sys
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


# utils/
from utils.MyColor import *
from utils.MyLogger import *
from utils.MyMongoDB import *

root_dataset_dirpath = "/data/users/gaoli/exp_Robust/datasets"
log_filename = "test.log"
# 使用的数据集名
dataset_dirname = "old"

blackset_dirname = "oldblack" # in dir[dataset_name]
whiteset_dirname = "oldwhite" # in dir[dataset_name]
blackset_dirpath = os.path.join(dataset_dir, dataset_dirname, blackset_dirname)
whiteset_dirpath = os.path.join(dataset_dir, dataset_dirname, whiteset_dirname)


blackset_dirpath = os.path.join(dataset_dir, dataset_dirname, blackset_dirname)
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

class Experiment(Operation):

    def __init__(self, logger: MyLogger = None, print_name: str = "｜Experiment「test」｜") -> None:
        Operation.__init__(self,logger,print_name)
        self.print_attention(f"实验 {print_name} 开始...")
    
    def _get_subset_dirpath(self, this_dataset_dirname:str, this_subset_dirname:str, root_dataset_dirpath:str = dataset_dir) -> str:
        '''
        get the dirpath of the subset
        '''
        subset_dirpath = os.path.join(root_dataset_dirpath,this_dataset_dirname, this_subset_dirname)
        return subset_dirpath
    

    def load_data(self, this_dataset_dirname:str, this_subset_dirname:str,root_dataset_dirpath:str = dataset_dir, onlyfromfile=False) -> List:
        
        '''
        load data from the subset
        path: dataset_dir/dataset_name/subset_name
        '''
        # 不止从文件中读取
        # # 数据库中存在， 从数据库中读取
        # # 否则， 从文件中读取
        if isexists_db_col(this_dataset_dirname, this_subset_dirname) and not onlyfromfile:
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
            pass
        # 从文件夹中读取
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

# def cross_entropy_clustering(ytrue:List, ypred:List)->float:
#     purity_each_cluster_b, purity_each_cluster_w, avg_purity_max_cluster, label_each_cluster = get_purity(ytrue, ypred)
#     # get the number of clusters
#     expected_purity = []
#     for each_num in purity_each_cluster_w:
#         if each_num > 0.5 + 0.01: # 更可能是white 蔟
#             expected_purity.append(1.)
#         elif each_num < 0.5-0.01:
#             expected_purity.append(0.)
#         else:
#             expected_purity.append(0.5)
#     # 计算purity_each_cluster_w 和 expected_purity 的交叉熵
#     expected_purity = np.array(expected_purity)
#     purity_each_cluster_w = np.array(purity_each_cluster_w)
#     print("---",purity_each_cluster_w, expected_purity)
#     cross_entropy=-np.sum(expected_purity * np.log(purity_each_cluster_w))
#     return cross_entropy

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
    onlyfromfile = False   # flag:是否只从文件读取数据
    lst_black_flows = exp.load_data(dataset_dirname, blackset_dirname,root_dataset_dirpath,onlyfromfile)
    lst_white_flows = exp.load_data(dataset_dirname, whiteset_dirname,root_dataset_dirpath,onlyfromfile)
    print(lst_black_flows[0].get_BagID())
    exp.print_info(f"黑样本流数量:{len(lst_black_flows)}")
    exp.print_info(f"白样本流数量:{len(lst_white_flows)}")
    # 提取包粒度的特征
    packetFeats = PacketFeatures(logger=mylogger)
    lst_pktFeats_black, lst_pktLabels_black = packetFeats.get_features_from_flows(lst_black_flows)
    exp.print_info(f"黑样本包数量：{len(lst_pktFeats_black)}")
    lst_pktFeats_white, lst_pktLabels_white = packetFeats.get_features_from_flows(lst_white_flows)
    exp.print_info(f"白样本包数量：{len(lst_pktFeats_white)}")
    # 取固定个数个packet

    num_pkt_each = int(input("输入聚类的黑白packet数目，推荐4000"))# 黑白packet各取num_pkt_each个packet
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
    # 画图设置
    fig_nrows = 6 # 4行 
    fig_ncols = 3 # 3列
    plt.figure(figsize=(fig_ncols*3.5+2, fig_nrows*3.5)) # 设置画布大小 宽=子图宽*列数+2，高=子图高*行数
    plt.subplot(fig_nrows, fig_ncols, 1)# 参数 fig_nrows, fig_ncols 分别指定了图像窗口中的子图行数和列数
        # 参数 1 指定了当前子图的位置，即第 1 个子图
    plt.title(f'real labels:{2} ', fontsize=14)
    plt.tick_params(labelsize=10) # 设置坐标轴刻度的字体大小
    
    # # tsne 二维映射
    starttime = time.time()
    
    X_umap= TSNE(n_components=2,init='pca', random_state=RANDOMSEED+1).fit_transform(np_pktFeats_all)
    # # UMAP
    # umap_model = umap.UMAP(n_neighbors=6, min_dist=0.3, metric='euclidean')
    # X_umap = umap_model.fit_transform(np_pktFeats_all)
    endtime = time.time()
    print("t-SNE: %.2g sec" % (endtime - starttime))
    # cmap
    # benigh
    # plt.scatter(X_tsne[:num_pkt_each,0], X_tsne[:num_pkt_each,1], s=2, marker="o", c='blue', cmap=mycmap)
    # attack
    # plt.scatter(X_tsne[:,0], X_tsne[:,1], s=0.1, marker="v", c=y_true, cmap=ListedColormap(c_list[:2]), alpha=alpha)
    plt.scatter(X_umap[:,0], X_umap[:,1], s=0.2, marker="v", c=y_true, cmap='tab20')
    # plt.legend(["Benigh","Malicious"],loc='upper right')
    # KMeans聚类
    for n_clusters in range(2, 15):
        # 8个子图
        model = KMeans(n_clusters=n_clusters,random_state=RANDOMSEED+1)
        # model = KernelKMeans(n_clusters=n_clusters, max_iter=100, kernel=pairwise.linear_kernel)
        # model = DBSCAN(eps=0.1, min_samples=10)
        # 预测标签
        y_pred = model.fit_predict(np_pktFeats_all)
        purity_each_cluster_b, purity_each_cluster_w, avg_purity_max_cluster, label_each_cluster = get_purity(np_pktLabels_all, y_pred)
        exp.print_info(f"每个簇的黑样本纯度:{purity_each_cluster_b}")
        exp.print_info(f"每个簇的白样本纯度:{purity_each_cluster_w}")
        exp.print_info(f"每个簇的最大纯度的平均值:{avg_purity_max_cluster}")
        exp.print_info(f"每个簇的最大纯度的标签:{label_each_cluster}")
        # 计算指标并输出
        metric = get_metric_PktClutering(np_pktLabels_all, y_pred)
        exp.print_attention(f"聚类的指标:{metric}")
        
        # 画图显示样本数据
        y_mapclass = get_mapclass_from_ypred(y_true, y_pred,threshold_black=0.51, threshold_white=0.56)
        # # subplot
        plt.subplot(fig_nrows, fig_ncols, n_clusters)
        # # 子图标签
        plt.title(f'N_clusters:{n_clusters} metric: {metric:.6f}', fontsize=14)
        # plt.xlabel('X', fontsize=14)
        # plt.ylabel('Y', fontsize=14)
        plt.tick_params(labelsize=10)
        # # 绘制散点
        plt.scatter(X_umap[:,0], X_umap[:,1], s=0.2, c=y_pred, cmap='tab20')
        # plt.legend()

    # 第九个图

    # plt.suptitle('Kmeans on Packet-level Features', fontsize=14)

    
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)


    plt.show()
    # 保存图片
    figname = f"kmeans_{9}.png"
    plt.savefig(os.path.join(figures_dir, figname),dpi=300)
    print("图片保存成功！",figname)


    switch_grain = Aggreator(logger=mylogger) # 聚合器
    lst_black_bags = switch_grain.aggregate(lst_black_flows)#  flow的lst -> bag的lst
    lst_white_bags = switch_grain.aggregate(lst_white_flows)


