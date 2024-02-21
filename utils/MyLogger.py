# -*- encoding: utf-8 -*-
'''
@File    :   MyLogger.py
@Time    :   2022/11/21 04:52:26
@Author  :   Hannah 
@Version :   2.0
@Contact :   1196033301@qq.com
@Desc    :   调试完毕！
'''

# import系统包
import logging
import os,sys

# import本级目录模块
this_filepath = __file__ # 此文件的路径: exp_Robust/utils/MyColor.py
this_dirpath = os.path.dirname(this_filepath) # 此文件所在文件夹/目录的路径: exp_Robust/utils
sys.path.append(this_dirpath)  
from MyColor import *

# import上级目录中的模块: 
# # 先将上级目录天加入环境变量，以便引入其他模块
parent_dirpath = os.path.join(this_dirpath, "..") # 加上..表示当前路径的上层目录:exp_Robust/
sys.path.append(parent_dirpath)  # 将上级目录添加到sys.path中，使得上级目录的模块可以被import
# # import 上级目录模块的config.py
from config import * 


# 默认日志设置：添加模式，日志文件名， 类名， 日志格式
dflt_filemode = 'a'
dflt_log_filename = 'default_logger.log'
dflt_name = "logger"
# default_format = '%(asctime)s %(levelname)s %(message)s'
# default_format1 = '%(name)-10s  %(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s'

class MyLogger:
    '''
    一个用于日志记录的类，可以将日志记录到文件和控制台。
    -----------
    Parameters:
    -----------
    name : str, optional (default='mylogger')
    logger的名称。
    log_filename : str, optional (default='mylogger.log')
        日志文件的文件名。
    filemode : str, optional (default='a')
        日志文件的写入模式，"a"为追加模式，"w"为覆盖模式。
    need_console : bool, optional (default=False)
        是否将日志输出到控制台。    

    Attributes:
    -----------
    logger : logging.Logger
        logging模块中的Logger对象。
    level : int
        日志的记录级别，默认为DEBUG级别。
    log_filepath : str
        日志文件的绝对路径。
    fmt : logging.Formatter
        日志信息的格式化器。

    Methods:
    --------
    get_logger():
        获取Logger对象。
    debug(msg: str):
        记录步骤信息。
    info(msg: str):
        记录重要信息。
    warning(msg: str):
        记录警告信息。
    error(msg: str):
        记录错误信息。
    critical(msg: str):
        记录严重错误信息。

    Examples:
    ---------
    1. append mode, 终端没有输出，样例：
    mylogger = MyLogger(name="test", log_filename="hello.log")
    2. write mode, 终端有输出
    mylogger = MyLogger(name="test", log_filename="test.log", filemode="w", need_console=True) 
   '''

   
    def __init__(self, name=dflt_name, log_filename=dflt_log_filename, filemode=dflt_filemode, need_console=False):
        self.logger = logging.getLogger(name=name)
        self.level = logging.DEBUG
        self.logger.setLevel(level=self.level)
        # handler
        self.log_filepath = os.path.join(log_dir, log_filename)
        f_hdlr = logging.FileHandler(filename=self.log_filepath, mode=filemode)
        f_hdlr.setLevel(level=self.level)
        c_hdlr = logging.StreamHandler()
        c_hdlr.setLevel(level=self.level)
        # formatter
        self.fmt = logging.Formatter(fmt="%(asctime)-21s"
                                    "%(levelname)-10s"
                                    "Msg: %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")
        f_hdlr.setFormatter(self.fmt)
        #f_hdlr设置filemode为a，即追加模式
        c_hdlr.setFormatter(self.fmt)
        self.logger.addHandler(hdlr=f_hdlr)
        if need_console:
            self.logger.addHandler(hdlr=c_hdlr)
        
        PrintColor().print_right(f"|MyLogger「{name}」|: \n\tcreate logfile : {os.path.abspath(self.log_filepath)}")
        

    def get_logger(self):# 获取logger：mylogger = MyLogger().get_logger()       
        return self.logger

    def debug(self, msg): # 描述步骤
        self.logger.debug(msg)

    def info(self, msg): # 描述重要信息
        self.logger.info(msg)

    def warning(self, msg): # 描述警报错误
        self.logger.warning(msg)

    def error(self, msg): # 描述错误
        self.logger.error(msg)

    def critical(self, msg): # 描述严重错误
        self.logger.critical(msg)


if __name__ == "__main__":
    mylogger = MyLogger(name="test", log_filename="test.log", filemode="w", need_console=True)
    mylogger.info("hello world")
    mylogger.debug("hello world")
