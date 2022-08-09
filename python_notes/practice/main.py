# 基本print指令
print("hello,张鹏举,it’s a new start." + " Hope everything is ok")
# 基本数学计算
# x = 3+2
# x = 3**2 #幂运算，表示3的平方
# x = 9//3 #向下取整
x = 9 % 2
print(x)
# 自变量 //命名规则，同其他语言
a, b, c = 1, 2, 3
print(a, b, c)
# 循环语句 while
i = 0
while i < 10:
    print(i)
    i += 1
# 循环语句 for
list1 = [1, 2, 3, 4, 5, 6, 7, 8]
for i in list1:
    print(i)
print("循环结束")
for i in range(1, 9, 3):  # range(start,end,step)函数,1~9开区间,步长3
    print(i)
# 判断语句 if
x, y, z = 1, 2, 3
if x > y > z:  # 运算符与C相同 > < >= <= == !=
    print("x<y<z")
elif x < y:  # 注:在if条件满足的时候，会立刻跳出整个if,哪怕后面也有满足条件的语句,也不再执行
    print("x<y")
elif y < z:
    print("y<z")
else:
    print("x>y>z")


# 函数定义
def add(a, b):
    print(a, b, a + b)
    return a + b


# 函数参数传入 1、按顺序依次写入 2、指定a= b=
add(1, 2)
add(a=4, b=2)


# 函数默认参数
def mult(length, width=10, height=5):  # 有默认参数的参数需在未定义默认值的参数后面
    print('length:', length, "width", width, "height", height)
    return length * width * height


mult(10, 5, 5)  # 默认参数是在未定义的情况下起作用，有定义时使用定义值
mult(10)
print(mult(10))
# 全局和局部变量 全局变量通常全部大写,类似于C中的宏定义
X = 100


def compare():
    d = 10
    return d


print(X)  # X 属于全局变量,在任意函数中均可识别
print(mult(X))
# print(d)          # d 属于compare函数中定义的变量,在外边识别不了

# 外部模块安装
# pip install ...

# 文件读写
text = "This is my first file operation.\nThis is a new line"
append_txt = "This is appended line"
print(text)
my_file = open("my_file.txt", 'w')  # open() 第一个参数 文件名 第二个参数 文件权限(w:写，r：读 ,a:append 追加）
my_file.write(text)
my_file.close()
my_file = open("my_file.txt", "a")
my_file.write(append_txt)
my_file.close()
file = open("my_file.txt", "r")
# content = file.read()           # 读文件的内容
content1 = file.readlines()  # 按行读文件,并存入一个列表
# print(content)
print(content1)


# 类
class Calculator:  # 类的首字母通常大写
    # name = "Good calculator"
    # price = 18

    def __init__(self, name=1, price=4):
        self.name = name
        self.pr = price
        self.add(1, 2)

    def add(self, x=1, y=2):
        return x + y


c = Calculator(50, 100)
print(c.name)
# input
# test_input = input("Please give a number:")  # 注： input() 返回值是字符串,判断时需要使用强制转换或者判断字符串
# print("This input number is :", test_input)
# 元组和列表
test_tuple = (1, 2, 3, 4, 5)  # 元组内数据不可更改
another_tuple = 2, 3, 4, 5, 6  # 列表里的数据可以更改

test_list = [5, 4, 3, 2, 1]
for index in range(len(test_list)):
    print("index = ", index, "number in list = ", test_list[index])
for index in range(len(test_tuple)):
    print("index = ", index, "number in list = ", test_tuple[index])
# 列表操作
test_list.append(0)
print(test_list)
test_list.insert(1, 0)  # 第一个参数是插入的位置，第二个参数是要插的数
print(test_list)
test_list.remove(0)  # 移除列表里第一个出现的这个值
print(test_list)
print(test_list.index(0))  # 输出第一次出现的这个值的索引值
print(test_list.count(0))  # 输出0这个值在列表中出现的次数
test_list.sort()  # 对列表进行排序,默认从小到大排,会覆盖掉原有列表 (改为从大到小需要使用reverse参数 reverse = True)
print(test_list)
test_list.sort(reverse=True)
print(test_list)
# 列表的索引
print(test_list[0])  # [0]输出列表第一个数
print(test_list[-1])  # [-1] 输出列表从后往前数第n个数 ，-1表示最后一个数
print(test_list[0:3])  # [0:3] 输出列表前3个数 ，0:3表示索引值从[0,3),左开右闭
print(test_list[:3])  # [:3] 索引值表示从开始到3,效果同上一条语句
print(test_list[3:])  # [3:] 索引值从3开始到最后
print(test_list[-3:])  # [-3:] 索引值从倒数第三个开始到最后
# 多维列表
mult_test_list = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]  # 更复杂的矩阵操作需要使用numpy库
print(mult_test_list[0][1])
# 字典
test_dictionary = {'apple': 1, 'pear': 2, 'orange': 3}  # 即键值对
print(test_dictionary['apple'])
del test_dictionary['pear']
print(test_dictionary)
# test_dictionary['banana'] = 2                            # 字典添加值的方法
# print(test_dictionary)
# test_dictionary['fruit'] = {'Strawberry': 4, 'peach': 5}  # 字典中加字典
# print(test_dictionary['fruit']['peach'])
d = Calculator()
test_dictionary['Calculator'] = d.add(5, 2)  # 字典中加类中的函数
print(test_dictionary['Calculator'])
test_dictionary['function'] = add(1, 2)  # 字典中加函数
print(test_dictionary['function'])  # 字典中可以添加任意东西,如列表，元组,函数,值，等等
# 载入模块  方法有34种
import time  # 直接引入
print(time.localtime())
import time as t  # 引入后重命名，适用于模块名字太长的模块简写
print(t.localtime())
from time import localtime  # 只引入time中的localtime函数
print(localtime())
from time import *  # 引入time模块中的所有函数
print(time)
import import_module as module
module.showdata()
# continue and break
# a = True
# while a:
#     b = int(input('input a number'))
#     if b == 1:
#         #continue
#         break
#     print('Already skip this loop')
# print('This position is out of loop')
# try
try:
    file = open('eeee.txt', 'r')
except Exception(BaseException) as e:
    print(e)
#     response = input("Do you want to create a new file?")
#     print('Press y to continue or Press n to exit')
#     if response == 'y':
#         file = open('eeee.txt', 'w')
#     else :
#         print('fine')
else:
    file.write(text)
file.close()
