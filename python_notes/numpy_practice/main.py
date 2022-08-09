import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
array的属性
array = np.array([[1, 2, 3], [2, 3, 4]])  # np.array() 将列表转换为矩阵
print(array)
print('number of dim', array.ndim)         # array的维数
print('shape of array', array.shape)       # array的形状，几乘几的矩阵
print('size', array.size)                  # array的元素个数
# 创建array
a = np.array([1, 2, 3], dtype=np.int64)    # dtype可以定义转换成矩阵之后的数据类型,包括int float 等
print(a.dtype)                             # 如果不指定精度的话可以不使用np.int64 直接dtype=int即可，np,int已经被弃用
a = np.zeros((3, 4), int)                  # 生成三行四列的零矩阵 第一个参数shape,第二个参数,数据类型
a = np.ones((3, 4), int)                   # 生成三行四列的一矩阵
a = np.empty((3, 4), dtype=np.float64)     # 生成接近于0的三行四列矩阵，数据类型为float64
a = np.arange(12).reshape((3, 4))          # 同python中的range ,reshape可以给矩阵变换形状
a = np.linspace(1, 10, 20)                 # 生成1~10,分成20小段后的数列
print(a)
# numpy基础运算
# ---------一维矩阵运算---------------------#
b = np.array([10, 20, 30, 40])
c = np.arange(4)
print(b, c)
print(b < 30)                              # 判断b中哪些元素小于30，打印True or False
d = b-c
d = c**2
print(d)
# ---------多维矩阵运算---------------------#
b = np.array([[1, 2], [3, 4]])
c = np.arange(4).reshape((2, 2))
d = b*c                                    # 逐个相乘
d_dot = np.dot(b, c)                       # 矩阵相乘
d_dot = b.dot(c)                           # 如果b是一个numpy,则可直接使用.dot()函数，效果同上一行
print(d, d_dot)
b = np.random.random((2, 4))               # 第一个random是模块，第二个random是函数 参数为shape
print(b)
print(np.sum(b, axis=1))                   # sum()求和函数，axis=1,求行的和
print(np.min(b, axis=0))                   # min()求最小值，axis=0,求列最小
print(np.max(b, axis=1))                   # max()求最大值，axis=1,求行最大
# ---------多维矩阵运算---------------------#
b = np.array(np.arange(2, 11).reshape((3, 3)))
print(b)
print(np.argmin(b))                         # 求b矩阵中最小值的索引
print(np.argmax(b))                         # 求b矩阵中最小值的索引
print(b.mean())                             # 求b矩阵的均值
print(np.median(b))                         # 求b矩阵的中位数
print(np.cumsum(b))                         # cumsum 累加，第n个元素为b矩阵前n个数之和
print(np.diff(b))                           # diff 累差，生成的矩阵每个元素为b矩阵相邻两个数之差
print(np.nonzero(b))                        # 输出所有非零元素的索引，分为两个array，
print(np.sort(b))                           # 逐行排序
print(np.transpose(b))                      # 矩阵转置,也可以用b.T来表示转置,T一定要大写
print(b.T)
# print(np.linalg.inv(b))                   # inv()求逆  因为生成的这个b矩阵行列式为0，不可逆，所以报错
print(np.clip(b, 5, 9))                     # 限幅滤波，将小于5的数全转换为5，大于9的数转换为0，中间的不变
# numpy索引
e = np.arange(2, 14).reshape((3, 4))
print(e[2][3])
print(e[2, 1])              # 输出第三行，第二个数
print(e[2, :])              # 输出第三行所有数
print(e[:, 1])              # 输出第二列所有数
print(e[1, 1:3])            # 输出第二行，第二到四列(开区间）

for row in e:               # for中，默认迭代e的行向量
    print(row)
for column in e.T:           # 迭代列可以使用转置
    print(column)
for item in e.flat:          # e.flat表示将3行四列矩阵转换为1行的，之后一个一个输出
    print(item)
# 合并array
f = np.array([1, 1, 1])[:, np.newaxis]
g = np.array([2, 2, 2])[:, np.newaxis]
print(np.vstack((f, g)))       # 把两个矩阵上下合并
print(np.hstack((f, g)))        # 把两个矩阵左右合并
print(f[:, np.newaxis])          # 把横向的矩阵转换为纵向的，与转置的区别是这是采用添加列的维度的方式转换的
h = np.concatenate((f, g, g, f), axis=0)
print(h)
# 分割array
i = np.arange(12).reshape((3, 4))
print(i)
print(np.split(i, 2, axis=1))     # np.spilt(),将矩阵分割，第一个参数是矩阵，第二个是分成几个小矩阵，第三个是方向，按行还是按列
# print(np.split(i, 4, axis=0))     # 直接这样是不能进行不等分割的，因为三行的数据，没法均分成4块，需要使用np.array_spilt()
print(np.array_split(i, 3, axis=1))   # 不均分，第一个array为两列，后面两个各一列，随机分的
print(np.vsplit(i, 3))                # 横向分割为3行
print(np.hsplit(i, 4))                # 纵向分割为4列
# copy和deepcopy
j = np.array([0, 1, 2, 3])
k = j
l = j
m = k
print(j.data,m.data)
print(j, k, l, m)
j[0] = 11                           # 赋值之后，一个改变，所有的都跟着改变
print(j, k, l, m)
m[1:3] = [22, 33]
print(j, k, l, m)
n = j.copy()                        # 解决办法，采用deep copy的方式
print(n)
print(n.data)
print(j.data)
l[3] = 44
print(j, k, l, m, n)

# pandas      相较于numpy，pandas更像是一个字典，他会给每个值添加一个序号
s = pd.Series([1, 2, 3, 4, np.nan])
print(s)
dates = pd.date_range("20220708", periods=6)
print(dates)
# 生成dataframe的方法
# 1.指定数据，指定行索引标签，列索引标签
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=['a', 'b', 'c', 'd']) # 生成一个6*4的矩阵，行索引为dates，列索引为a,b,c,d
print(df)
# 2.直接放一个字典进去
df2 = pd.DataFrame({'A': 1., 'B': pd.date_range("20220708", periods=3), 'C': np.array([11, 12, 13])})
print(df2)
# dataframe的属性
print(df2.dtypes)            # 输出列数据类型
print(df2.index)             # 输出行索引的值
print(df2.columns)           # 输出列索引的值
print(df2.values)            # 输出矩阵的值
print(df2.describe())        # 输出矩阵的统计信息，包括个数，平均值，标准差，最大值等，只能计算数字，不能计算字符，日期数据
print(df2.T)                 # 转置
print(df2.sort_index(axis=1, ascending=False))  # 按列标签排序，ascending=False表示倒序
print(df2.sort_index(axis=0, ascending=False))
print(df2.sort_values(by='C', ascending=False))  # 按值对第3列进行排序，ascending=False表示倒序
# ******************************************************************************************************
# pandas 选择数据
# 1.直接选择
print(df['a'], df.a)         # 这两种表达方式是一样的
print(df[:3], df['2022-07-08':'2022-07-10'])  # 这两种表达方式是一样的
print(df[2:][0:1])
# 2. select by label:loc
print('df.loc:', df.loc['2022-07-08'])
print('df.loc1', df.loc[:, ['a', 'b']])            # df.loc() []中为行和列，[:,['a', 'b']]表示取所有行,ab两列
# 3. select by position: iloc
print(df.iloc[3:, 1])                              # df.iloc() []为行和列，其中[3:,1]表示第4行到最后，第一列的数
# 4.Boolean indexing
print(df)
print(df[df.a > 0])                                # 输出a这一列中大于0的所有行
# ******************************************************************************************************
# 改变指定位置的值
df.iloc[2, 3] = 1111
df.loc['2022-07-08', 'a'] = 2222
df.a[df.a > 0] = 3333
df['f'] = np.nan                                    # 添加一个空行
print(df)
# 处理不完整或丢失的数据
print(df.dropna(axis=1,how = 'any'))                # 第一个参数，按行丢弃或者按列丢弃,第二个参数：how,为any时，只要有一个nan则丢弃整行
# print(df.fillna(value=0))                           # 将nan填充为0
print(df.isnull())                                   # 判断是否缺失数据
print(np.any(df.isnull()))                           # 判断整个矩阵里有没有丢失数据
# 导入导出数据        支持csv excel hdf sql json 等等格式数据
data = pd.read_csv('student.csv')
print(data)
data.to_json('test2.json')
# pandas 合并
# concat合并
# 相同列合并，上下合并
df3 = pd.DataFrame(np.ones((3, 4))*0, columns=['a', 'b', 'c', 'd'])
df4 = pd.DataFrame(np.ones((3, 4))*1, columns=['a', 'b', 'c', 'd'])
df5 = pd.DataFrame(np.ones((3, 4))*2, columns=['a', 'b', 'c', 'd'])
res = pd.concat([df3, df4, df5], axis=0)
print(res)                           # 输出结果中行索引值不连续，是单纯的将他们合并起来而已，解决办法：参数ignore_index=True
res = pd.concat([df3, df4, df5], axis=0,ignore_index=True)
print(res)
# 列索引部分重叠
df3 = pd.DataFrame(np.ones((3, 4))*0, columns=['a', 'b', 'c', 'd'],index=[1,2,3])
df4 = pd.DataFrame(np.ones((3, 4))*1, columns=['b', 'c', 'd', 'e'],index=[2,3,4])
res = pd.concat([df3, df4])
print(res)                                 # 合并效果，没有的值用nan填充，是因为concat默认的是使用outer模式
res = pd.concat([df3, df4], join='inner', ignore_index=True)
print(res)                                 # 改为inner模式后，只把相同的部分合并在一起，其余的舍去
# 左右合并，行索引不同
res = pd.concat([df3, df4], axis=1)
print(res)
# merge方法合并 默认模式是inner
# 有其中一列相同的情况下
df3 = pd.DataFrame({'a': ['a0','a1','a2','a3'], 'b': ['b0', 'b1', 'b2', 'b3'], 'k': ['k0', 'k1', 'k2', 'k3']})
df4 = pd.DataFrame({'c': ['c0','c1','c2','c3'], 'd': ['d0', 'd1', 'd2', 'd3'], 'k': ['k0', 'k1', 'k2', 'k3']})
print(df3,df4)
res = pd.merge(df3,df4, on='k')                # df3,df4，按k这一列合并
print(res)
# 两列的列索相同，但是内容不同
df3 = pd.DataFrame({'a': ['a0', 'a1', 'a2', 'a3'], 'b': ['b0', 'b1', 'b2', 'b3'], 'k1': ['k0', 'k0', 'k1', 'k1'], 'k2': ['k0', 'k1', 'k0', 'k1']})
df4 = pd.DataFrame({'c': ['c0', 'c1', 'c2', 'c3'], 'd': ['d0', 'd1', 'd2', 'd3'], 'k1': ['k0', 'k1', 'k1', 'k0'], 'k2': ['k0', 'k0', 'k0', 'k0']})
print(df3, df4)
res = pd.merge(df3, df4, on=['k1', 'k2'])                # df3,df4，按k1,k2这两列合并，默认是inner模式，只保留相同的k1，k2都相同的项
print(res)
# 画图plot
# data = pd.Series(np.random.randn(1000), index=np.arange(1000))
# data = data.cumsum()
data = pd.DataFrame(np.random.randn(1000,4),index= np.arange(1000),columns=list("ABCD"))
data = data.cumsum()
print(data.head())
ax=data.plot.scatter(x='A', y='B', color='Blue', label='class 1')
data.plot.scatter(x='A', y='C', color='Green', label='class 2', ax=ax)
# data.plot()
plt.show()











