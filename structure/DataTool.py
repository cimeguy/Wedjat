# -*- encoding: utf-8 -*-
'''
@File    :   DataTool.py
@Time    :   2022/11/10 01:34:14
@Author  :   Hannah 
@Version :   1.0
@Contact :   1196033301@qq.com
@Desc    :   更底层的工具
'''

# here put the import lib

from typing import *

import os,sys
# 不能引入Flow、Packet等，会循环引用

# 将BetweenFlow dir 添加入环境变量, 以便导入其他
this_filepath = __file__ # 此文件的路径
this_dirpath = os.path.dirname(this_filepath) # 此文件所在文件夹/目录的路径
parent_dir = os.path.join(this_dirpath, "..") # 加上..表示当前路径的上层目录
sys.path.append(parent_dir)  # 将上级目录添加到sys.path中，使得上级目录的模块可以被import
from config import * # 必须提前导入配置文件
from utils.MyMongoDB import *
from utils.MyColor import *


# 一些标识
OBJ_DctPkt = "Object_DictPacket"
OBJ_DctFlw = "Object_DictFlow"
ObJ_DctBag = "Object_DictBag"
## 包方向
DIR_src2dst = 1
DIR_dst2src = -1


'''
    fivetuple:List[str] = ["src_ip","dst_ip","src_port","dst_port","protocol"]
'''

## for test 
### 5tuple 区分packet
### 5tuple, belongfile 区分flow
test1_5tuple = ["A_ip","B_ip","A_port","B_port","TCP"]
test1_3tuple = ["A_ip","B_ip","data/pcap/1.pcap"]
test1_2tuple = ["A_ip","B_ip"]


# 判断该字典数据是Flow还是Packet
def isinstance_dict(dict_data: Dict):
    if "type_data" in dict_data.keys():
        if dict_data["type_data"] == "Packet":
            return OBJ_DctPkt
        elif dict_data["type_data"] == "Flow":
            return OBJ_DctFlw
        elif dict_data["type_data"] == "Bag":
            return ObJ_DctBag
        else:
            return None
    if "origin" in dict_data.keys():
        return OBJ_DctPkt
    elif "lst_packets" in dict_data.keys():
        return OBJ_DctFlw
    elif "lst_flows" in dict_data.keys():
        return ObJ_DctBag
    else:
        return None
    