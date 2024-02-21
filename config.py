# -*- encoding: utf-8 -*-
'''
@File    :   config.py
@Time    :   2022/11/21 03:47:41
@Author  :   Hannah 
@Version :   1.0
@Contact :   1196033301@qq.com
@Desc    :   配置文件，配置文件目录
'''

# here put the import lib


# here put the import lib
import os, sys

# 工作路径
# pwd = sys.path[0] # 当前正在运行的文件所在的目录

current_filepath = __file__ # 此文件的路径
current_dirpath = os.path.dirname(current_filepath) # 此文件所在文件夹/目录的路径
pwd = current_dirpath # 工作目录

# pcap数据集所在文件夹的路径
dataset_dir = os.path.join(pwd,"datasets")
# label
UNKNOWN = "unknown"
BLACK = 1
WHITE = 0


RANDOMSEED = 2023
# 子目录
# structure 数据结构类文件所在目录/文件夹的路径
structure_dir = os.path.join(pwd, "structure")
sys.path.append(structure_dir)  # ‼️ 使structure内部文件互相引用
# utils 实验工具文件所在目录的路径
utils_dir = os.path.join(pwd, "utils")
sys.path.append(utils_dir)  # ‼️ 使utils内部文件互相引用

# log 日志文件所在目录的路径
log_dir = os.path.join(pwd, "log")

# figures 图片文件所在目录的路径
figures_dir = os.path.join(pwd, "figures")


