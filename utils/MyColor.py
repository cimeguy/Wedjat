# -*- encoding: utf-8 -*-
'''
@File    :   MyColor.py
@Time    :   2022/11/21 04:52:41
@Author  :   Hannah 
@Version :   2.0
@Contact :   1196033301@qq.com
@Desc    :   调试完毕！
'''

# import系统包
from colorama import  init,Fore


# 该类不直接使用
# 设置字符串输出格式：选择的颜色+string+恢复默认颜色
class Colored(object):
    def __init__(self) -> None:
        init(autoreset=True)

    #  前景色:红色  背景色:默认
    def red(self, s):
        return Fore.RED + s + Fore.RESET
    
    #  前景色:绿色  背景色:默认
    def green(self, s):
        return Fore.GREEN + s + Fore.RESET

    #  前景色:黄色  背景色:默认
    def yellow(self, s):
        return Fore.YELLOW + s + Fore.RESET

    #  前景色:蓝色  背景色:默认
    def blue(self, s):
        return Fore.BLUE + s + Fore.RESET
    
    # 前景色：品红 背景色：默认
    def magenta(self, s):
        return Fore.MAGENTA + s + Fore.RESET
    
    # 前景色:蓝绿色  背景色:默认
    def cyan(self, s):
        return Fore.CYAN + s + Fore.RESET
    
    #  前景色:白色  背景色:默认
    def white(self, s):
        return Fore.WHITE + s + Fore.RESET

    #  前景色:黑色  背景色:默认
    def black(self, s):
        return Fore.BLACK


    
# 打印颜色： 通常被PrintTool类包含
class PrintColor():
    # 打印的颜色模式
    def __init__(self) -> None:
        self.color = Colored()

    # private function
    # 为错误操作的字符串赋予颜色： 红色
    def _color_wrong(self, str):
        return self.color.red(str)
    
    # 为正确操作的字符串赋予颜色： 绿色
    def _color_right(self, str):
        return self.color.green(str)
    
    # 为有用「信息」的字符串赋予颜色： 黄色
    def _color_info(self, str):
        return self.color.yellow(str)
    
    # 为不是特别重要的「信息」字符串赋予颜色： 蓝绿色
    def _color_NI_info(self, str):
        return self.color.cyan(str)
    
    # 为需要注意的「信息」字符串赋予颜色： 品红色
    def _color_attention(self,str):
        return self.color.magenta(str)

    # public function
    # 正常打印，不带颜色
    def print(self,str, end = '\n'):
        print(str,end)

    # 打印错误的操作
    def print_wrong(self, str):
        print(self._color_wrong(str))
    def print_warning(self,str):
        self.print_wrong(str)

    # 打印正确的操作
    def print_right(self, str):
        print(self._color_right(str))

    # 打印信息的操作
    def print_info(self, info):
        print(self._color_info(info))

    # 打印注意信息 
    def print_attention(self,str):
        print(self._color_attention(str))

    # 打印不重要的信息
    def print_NI_info(self,str,end = '\n'):
        print(self._color_NI_info(str),end)



# 测试
if __name__=="__main__":
    
    PrintColor().print_wrong("错误的操作")
    PrintColor().print_right("正确的操作")
    PrintColor().print_info("信息")
    PrintColor().print_attention("注意")
    PrintColor().print_NI_info("不重要的信息" ,end = '12')
    PrintColor().print("正常打印")
    name = "hannah"
    print(f"{Fore.RED} + {name} +{Fore.RESET}",end="")

