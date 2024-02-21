# -*- encoding: utf-8 -*-
'''
@File    :   splitDatasets.py
@Time    :   2022/11/26 21:34:26
@Author  :   Hannah 
@Version :   1.0
@Contact :   1196033301@qq.com
@Desc    :   description...
'''

# here put the import lib

# sys import
import os, sys, shutil, random
from tqdm import tqdm
# here put the import lib
from preProcess import *
from structure.Flow  import *
from structure.Packet import *
from structure.Bag import *
from ABCClass import Operation
# utils/
from utils.MyColor import *
from utils.MyLogger import *
# 设置随机种子
random_seed = 42
random.seed(random_seed)


class SplitDatasets():
    '''
    Operation SplitDatasets
    --------------------------------
    Paramters:
    logger: MyLogger
    '''
    def __init__(self) -> None:
    
        
        # pass
        self.all_black = ''
        self.all_white = ''
        self.train_black = ''
        self.train_white = ''
        self.test_black = ''
        self.test_white = ''

    def get_statistics(self, path_dataset):
        lst_files = ReadFile().get_allfile(dirpath=path_dataset)
        print(len(lst_files))


    def clearfolder(self, path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
            else:
                pass

    def copy(self, src_dir, dst_tr_dir, dst_te_dir, num_tr, num_te):
         # clear dst_tr_dir
        print(f"clear{dst_tr_dir} and {dst_te_dir}")
        self.clearfolder(dst_tr_dir)
        self.clearfolder(dst_te_dir)
        lst_files = ReadFile().get_allfile(dirpath=src_dir)
        # 随机打乱
    
        # 生成新的随机打乱的列表
        shuffled_list = random.sample(lst_files, len(lst_files))
        # copy
        lst_files = shuffled_list[:num_tr+num_te]
        
        train_files = lst_files[:num_tr]
        test_files = lst_files[num_tr:]
        print(len(lst_files))
        print(f"copy {len(train_files)} files to {dst_tr_dir}")
        print(f"copy {len(test_files)} files to {dst_te_dir}")
        for file in tqdm(train_files):
            shutil.copy(file, dst_tr_dir)
        for file in tqdm(test_files):
            shutil.copy(file, dst_te_dir)
        self.get_statistics(path_dataset=dst_tr_dir)
        self.get_statistics(path_dataset=dst_te_dir)
    


if __name__ == '__main__':
    SD = SplitDatasets()
    black = '/data/users/gaoli/exp_Robust/datasets/Private/black_21'
    dst_tr_black = '/data/users/gaoli/exp_Robust/datasets/118/train_black'
    dst_te_black = '/data/users/gaoli/exp_Robust/datasets/118/test_black'

    white = '/data/users/gaoli/exp_Robust/datasets/Private/white0'
    dst_tr_white = '/data/users/gaoli/exp_Robust/datasets/118/train_white'
    dst_te_white = '/data/users/gaoli/exp_Robust/datasets/118/test_white'
    SD.get_statistics(path_dataset=black)
    SD.copy(src_dir=black, dst_tr_dir=dst_tr_black, dst_te_dir=dst_te_black, num_tr=int(input("训练文件数量")), num_te=int(input("测试文件数量")))
    SD.get_statistics(path_dataset=white)
    SD.copy(src_dir=white, dst_tr_dir=dst_tr_white, dst_te_dir=dst_te_white, num_tr=int(input("训练文件数量")), num_te=int(input("测试文件数量")))