
# -*- encoding: utf-8 -*-
'''
@File    :   MyMongoDB.py
@Time    :   2022/11/21 04:51:49
@Author  :   Hannah 
@Version :   2.0
@Contact :   1196033301@qq.com
@Desc    :   带颜色的输入输出
                调试完毕！
'''

# import系统包
import pymongo   
from typing import *
import datetime,time
import os,sys

# import上级目录中的模块:
# # 将上级目录加入环境变量，以便引入其他模块
this_filepath = __file__ # 此文件的路径
this_dirpath = os.path.dirname(this_filepath) # 此文件所在文件夹/目录的路径
parent_dirpath = os.path.join(this_dirpath, "..") # 加上..表示当前路径的上层目录
sys.path.append(parent_dirpath)  # 将上级目录添加到sys.path中，使得上级目录的模块可以被import

# import本级目录中的模块
from MyColor import *



# 设置
# # mongodb数据库的路径
MONGODBPATH = "mongodb://localhost:27017/"



# 判断数据库db_name中是否存在某个集合collection_name
def isexists_db_col(db_name:str, collection_name:str):
    myclient = pymongo.MongoClient(MONGODBPATH)
    dblist = myclient.list_database_names()
    if db_name not in dblist:  # 不存在数据库db_name
        print(f"数据库{db_name}不存在！")
        return False
    mydb = myclient[db_name]   # 存在数据库db_name
    collist = mydb.list_collection_names()
    if collection_name in collist: # 存在集合collection_name
        return True
    else: # 不存在集合collection_name
        print(f"数据库{db_name}下的集合{collection_name}不存在！")
        return False



class MyMongoDB:   
    # 创建数据库和对应的集合
    def __init__(self, database, collection):   
        '''
        创建数据库和对应的集合

        Parameters:
        ----------
        database: str
            数据库名
        collection: str
            集合名
        
        ''' 
        # 连接到数据库、创建集合
        self._connet = pymongo.MongoClient(MONGODBPATH)   #连接到虚拟机
        self.database = self._connet[database] #连接到数据库
        self.collection = self.database[collection]  # 连接到集合（如果集合名不存在会自动创建）
        # 设置print
        self.database_name = database
        self.collection_name = collection
        self.print_name = f"|DB「{self.database_name}:{self.collection_name}」|"
        self.pc = PrintColor()


    def delete(self, query: Dict=None): 
        '''
        Example:
            mydb.delete({"five_tuple":[1,2,3,4,5]})
        '''
        #删除数据，query：需要删除的数据
        x = self.collection.delete_many(query)  #删除多条

    def delete_all(self): 
        '''
        Example:
            mydb.delete_all()
        '''
        # 删除该集合中所有数据
        x = self.collection.delete_many({})    #删除集合中所有数据，返回删除的数据条数
        print("\t",self.print_name,": ",x.deleted_count, "个文档已删除(所有文档)")

    def clear(self): 
        '''
        Example:
            mydb.clear()
        '''
        #清空集合中的所有数据
        self.pc.print_right(f"{self.print_name}: success! clear all data.")
        self.delete_all()

    def insert(self, data, ordered=True): 
        #输入需要添加的数据（data),ordered控制是否按顺序添加
        '''
        insert data into collection

        Parameters:
        ----------
        data: dict or List[dict] or else
            dict: one data or 
            list[dict]: many data or
            else 
                no operation
        ordered: bool=True
            if True 
                insert data in order, 
            else 
                insert data unordered
        '''
        if isinstance(data, Dict):# 表示单条数据， 一个Dict
            self.collection.insert_one(data)
        elif (isinstance(data, List) and len(data)==1): # 表示单条数据形成的一个列表: List[Dict]
            self.collection.insert_one(data[0]) 
        elif isinstance(data, List) and len(data)>1: # 表示多条数据构成的列表
            self.collection.insert_many(data,ordered=ordered) # ordered=False表示不按顺序插入，提高插入速度
        else:
            pass

    def find(self, query: Dict= None) -> List:  
        #输入查询的条件，并转换成列表, 返回查询结果
        '''
        Example:
            # 查找数据类型为black、 数据集类型为train的数据  
            for i in mydb.find({"datatype":"black","datasettype":"train"}):
                print(i)
        '''
        # query={"datatype":"flow","dataset_type":"train", "label":"black"} 训练集的黑样本
        result = list(self.collection.find(query))
        # print(len(result),"个文档已找到")    
        return result      
    
    def updata(self, data: Dict, new_data: Dict, onlyone=False): 
        # 指定需要修改的数据（data），修改后的数据（new_data）,onlyone控制修改单条还是多条: True, 修改单条数据
        # 可以更改已经存在的字段或添加新的字段
        '''
        Example:  
            # 更新数据：添加age字段
            print("update data:",{ 'age': 0})
            mydb.updata(data={'datatype': 'black'}, new_data={ 'age': 0}, onlyone=False)
        '''
        if onlyone:  #当onlyone为真
            self.collection.update_one(data, {'$set': new_data}) #修改单条数据，使用'$set'表示指定修改数据否则会使数据库中所有数据被新数据覆盖
        else:#当onlyone为假
            self.collection.update_many(data, {'$set': new_data}) #修改多条数据，使用'$set'表示指定修改数据否则会使数据库中所有数据被新数据覆盖

    def drop_collection(self):
        # 删除集合
        '''
        Example:
            mydb.drop_collection()
        '''
        self.pc.print_right(f"drop datasets {self.print_name}")
        self.collection.drop()  #删除集合

    def cnt_all(self): 
        # 统计集合中数据的条数
        '''
        Example:
            mydb.cnt_all()
        '''
        return self.collection.count_documents({})
       
    def isempty(self)->bool: 
        # 判断集合是否为空
        '''
        Example:
            bool_empty = mydb.isempty()
        
        '''
        return self.cnt_all()==0
        
    def print_front(self, num: int=10): 
        #打印集合中前num条数据
        '''
        Example:
            print("find data:")
            mydb.print_front(5) #打印前5条数据
        '''
        print("print first {} data:".format(num))
        result = self.collection.find().limit(num)
        for i,fetch in enumerate(result):
            print("No.",i)
            print(fetch)
        self.pc.print_info(f"__________{self.print_name} has {self.cnt_all()} data now__________\n")



if __name__ == "__main__":

    #指定我们需要操作的数据库为test数据库，需要操作的集合为test_flow集合
    mydb = MyMongoDB('test_for_me',"test_230508")  

    #删除wl集合中所有的数据
    mydb.clear() 

    # 插入数据
    mydb.insert([
        {'5tuple': [1,2,3], 'label': 'black', 'type_dataset':"train", "type_data":"flow", 
         "desc": "after evasion"},
        {'5tuple': [1,0,3], 'label': 'black', 'type_dataset':"train", "type_data":"bag", 
         "desc": "source_flow"},
        {'5tuple': [2,1,4], 'label': 'black', 'type_dataset':"test",  "type_data":"bag",
         "desc": "filter_flow"}
    ])

    # 打印前10条数据
    print("print first 10 data:")
    mydb.print_front() 

    # 更新数据：添加age字段
    print("update data:",{ 'age': 0})
    mydb.updata(data={'label': 'black'}, new_data={ 'age': 0}, onlyone=False)
    
    # 查找数据
    print("find data:")
    for i in mydb.find({"label":"black","type_dataset":"train"}):
        print(i)
    