baselines:其他论文的检测模型
datasets:pcap数据文件，下一层目录按照数据集名划分，再下一层按照黑白样本划分
evasion:存放各种evasion策略文件
R2E:我的检测模型
log:日志目录
# 实验主体部分
main.py:运行实验
‘’$setname = dataset_name + exp_desc(对实验的描述)‘’
1. preProcess.py: datasets -> database[$setname:original_flow]
2. filter.py: original_flow -> database[$setname:$filterRule_flow]
3. aggregate.py: $xxx_flow -> database[$setname:$xxx_bag]
4. splitSet.py: $xxx_bag -> database[$setname:$xxx_bag_trb,xxx_bag_tre,xxx_bag_trw, xxx_bag_tew]
5. evade.py: $xxx_bag_teb -> $xxx_bag_teb_ename_params
6. extractFeats.py: $xxx_bag_xxx -> detection/methodA/xxx/xxx.xx
7. detect.py: xxx.xx->predict results
8. calcul_metrics.py: predict results->metrics


# tools
config.py:配置文件
ABCClass.py: 操作类、打印类、数据存储类
utils:运行试验时需要的工具：颜色、日志、数据库



# 更新240118
splitDatasets.py 在开始前先划分数据集

main_bn copy.py 运行
<!-- SVC_ByesOptim_：核聚类，贝叶斯优化，
其中，kernelKmeans来自于https://tslearn.readthedocs.io/en/stable/gen_modules/clustering/tslearn.clustering.KernelKMeans.html -->
<!-- heatmap.ipynb: 绘制流量的评分热力图，内容是正常流量、异常流量、逃逸的异常流量在包评分映射后的Graph的热力图展示
Parameters_and_BP.ipynb： 参数学习和BP推断 -->

