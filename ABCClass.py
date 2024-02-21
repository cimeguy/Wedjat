# -*- encoding: utf-8 -*-
'''
@File    :   MyClass.py
@Time    :   2022/11/18 15:27:35
@Author  :   Hannah 
@Version :   1.0
@Contact :   1196033301@qq.com
@Desc    :   PrintTool, DataManager, Operation
                调试完成

'''

# here put the import lib
from config import *

# here put the import lib from sub direction
from utils.MyLogger import *
from utils.MyColor import *


width = 20
class PrintTool:
    '''
    为数据结构打印不同颜色的输出工具，通常被包含在数据结构或者其他类中
    ------------------------------
    该类的属性为函数
    ------------------------------
    example:
        pc = PrintColor()
        print_wrong = pc.print_wrong
        print_info = pc.print_info
        print_right = pc.print_right
        print = pc.print
        ------------------------------
        print_wrong("wrong")
        print_info("info")
        ...
        print_wrong("wrong infomation")

    '''
    def __init__(self, print_color: PrintColor = PrintColor()) -> None:
        # 设置输出颜色的模式
        self.pc = print_color
        self.print_info = self.pc.print_info # func
        self.print_wrong = self.pc.print_wrong
        self.print_right = self.pc.print_right
        self.print_attention = self.pc.print_attention
        self.print_NI_info = self.pc.print_NI_info
        self.print = print

class DataManager:
    '''
    # 数据集处理器
    DataManager class:
    has data and can operate on data 
        store from mongodb
        store to mongodb
    ------------------------------
    Paramters:
        logger: MyLogger object = None
            for logging info
        print_name: str = "｜Operation「name」｜"
            operation name
    ------------------------------
    Example:
        mylogger = MyLogger(name="test", log_filename="test.log") 
        dm = DataManager(logger=mylogger, print_name="|PktMngr「{name}」|")
        dm.print_wrong("wrong")
    or:
        DataManager.__init__(self, logger, print_name=f"|PktMngr「{name}」|" )
    ''' 
    def __init__(self,logger: MyLogger=None, print_name: str="|DataMngr「store_test」|") -> None:
        
        
        self.print_name = f"{print_name}" # 用于输出的名字

        self.pc= PrintColor()
        self.print_wrong = self.pc.print_wrong
        self.print_info = self.pc.print_info
        self.print_right = self.pc.print_right
        self.print =self.pc.print
        self.print_right(f"{self.print_name}: init! ")
        self.logger = logger
        
        if logger == None:
            print(f"logger{self.print_name} is None")
            self.logger = MyLogger().get_logger()
            self.logger.debug(f"{self.print_name} has no logger, use default logger")
            self.pc.print_NI_info(f"{self.print_name} has no logger, use default logger")
        else:
            self.logger.debug(f"{self.print_name} start")
        self.log_info = self.logger.info
        self.log_debug = self.logger.debug
        self.log_error = self.logger.error
        self.log_warning = self.logger.warning
        self.log_critical = self.logger.critical
        self.op_name ="" # operation name : need define in subclass

class Operation:
    '''
    Operation abstract class:
    log info and print info, not has or save data
    ------------------------------
    Paramters:
        logger: MyLogger object = None
            for logging info
        print_name: str = "｜Operation「name」｜"
            operation name
    ------------------------------
    Example:
        mylogger = MyLogger(name="test", log_filename="test.log")
        op = Operation(logger=mylogger, print_name="|FlwMngr「{name}」|")
        or: 
        Operation.__init__(self, logger, print_name=f"|PktMngr「{name}」|" )
    ''' 
    def __init__(self, logger:MyLogger=None, print_name:str="｜Operation「name」｜") -> None:
        
        # print 部分
        self.pc= PrintColor()
        self.print_wrong = self.pc.print_wrong
        self.print_info = self.pc.print_info
        self.print_right = self.pc.print_right
        self.print_attention = self.pc.print_attention
        self.print =self.pc.print
        self.print_right(f"{print_name}: init! ")
        self.logger = logger
        self.print_name =print_name # 用于输出的名字
        
        # log 部分
        if logger == None:
            print(f"logger{self.print_name} is None")
            self.logger = MyLogger().get_logger()
            self.logger.debug(f"{self.print_name} has no logger, use default logger")
            self.pc.print_NI_info(f"{self.print_name} has no logger, use default logger")
        else:
            self.logger.debug(f"{self.print_name} start")
        self.log_info = self.logger.info
        self.log_debug = self.logger.debug
        self.log_error = self.logger.error
        self.log_warning = self.logger.warning
        self.log_critical = self.logger.critical
        self.op_name ="" # operation name: : need define in subclass


if __name__ == "__main__":
    print("test PrintTool:")
    pt = PrintTool()
    pt.print_wrong("wrong")
    pt.print_attention("attention")
    pt.print_right("right")
    print("\ntest DataManager:")
    dm = DataManager()
    dm.print_info(dm.print_name)
    print("\ntest Operation:")
    mylogger = MyLogger(name="test",log_filename="test.log")
    op = Operation(logger=mylogger,print_name="｜OP「test_OP1」｜")
    op.print_info(op.print_name)
    op = Operation(print_name="|OP「test_OP2」|")
    op.print_info(op.print_name)