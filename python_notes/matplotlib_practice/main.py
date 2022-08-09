import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

x = np.linspace(-3, 3, 50)
y = x ** 3
z = 2 * x ** 2 + 1
# y1 = 0.1*x
# figure()下面的所有的图都属于一个figure里，直到出现下一个figure
# plt.figure(num=1, figsize=(8, 5))    # 参数说明：num表示figure标号,figsize表示长宽，（8，5）表示800*500像素的
# plt.plot(x, y)
# plt.figure(num=2)
# plt.plot(x, y)
# plt.plot(x, z, color="red", linewidth=4.0, linestyle='--') # 参数说明，color可以改变颜色，linewidth表示线型，linestyle表示实线虚线等
# plt.show()

# 图片坐标轴设置
plt.figure(num=1, figsize=(8, 5))  # 参数说明：num表示figure标号,figsize表示长宽，（8，5）表示800*500像素的
plt.plot(x, y, lw=20, zorder=1)
# plt.xlim((-1, 3))                  # 横坐标显示-1~0.5之间的数
# plt.ylim((-1, 5))                  # 纵坐标显示-0.5~1之间的数
plt.xlabel('x', loc='right')  # 横轴名称 x
plt.ylabel('y', loc='top')  # 纵轴名称 y
new_sticks = np.linspace(-1, 1, 5)
plt.xticks(new_sticks)
plt.yticks([-1, 0, 1], [r'$really\ bad$', r'$\ \alpha$', 'really good'])
# 将特定的数与特定的标签对应,
# r''表示正则化,
# $really\ bad$表示用数学方式显示中间的字符，中间的\+空格表示转义空格，目的是在图中显示空格
# r'$\ \alpha$ 表示将alpha转义为数学表达式中的a，跟LaTex语法类似

# gca = get current axis
ax = plt.gca()  # 获取当前的轴
ax.spines['right'].set_color('none')  # 消除图中右边框
ax.spines['top'].set_color('none')  # 消除图中上边框
ax.xaxis.set_ticks_position('bottom')  # 设置x轴为下边的轴
ax.yaxis.set_ticks_position('left')  # 设置y轴为右边的轴
ax.spines['bottom'].set_position(('data', 0))  # 设置x轴位置为纵轴值为0的那一行
ax.spines['left'].set_position(('data', 0))  # 设置y轴位置为纵轴值为0的那一行

# 设置图例
# l1, = plt.plot(x, y, label='up')
# l2, = plt.plot(x, z, color="red", linewidth=4.0, linestyle='--',label='down')
# plt.legend(handles=[l1,l2],labels=['aaa','bbb'],loc='best')
# plt.legend(),用handles参数时，需要将plot赋给一个变量，并且加逗号，如上所示。
# labels参数：修改之前设定好的图例参数，如之前设置的是up和down，在这里更改之后就变成aaa,bbb了
# loc参数：图例放置的位置，best表示自动根据曲线调整最佳位置

# 在曲线上进行标注 annotation
# x0 = 0.5
# y0 = 2*x0**2+1
# plt.scatter(x0, y0,s=50,color='g')            # 将这个点标注出来，颜色为绿色，大小为50
# plt.plot([x0, x0], [y0, 0], 'k--', lw=2.5)
# plot可以根据给定的两点坐标来画图，对应关系是,(x0,y0),(x0,0)两点的连线
# k--是一个缩写，k表示黑色，--是线型为虚线，lw是线的宽度，2.5
# 注释方法一
# plt.annotate(r'$2x^2+1=%s$'%y0,xy=(x0,y0),xycoords='data',xytext=(+30,-30),textcoords='offset points',
#             fontsize=16,arrowprops=dict(arrowstyle="->",connectionstyle='arc3,rad=0.2'))
# 参数说明：
# r'$2x**2+1=%s$'%y0正则化表达式，与之前的结果类似，%s跟c语言的print类似，
# 后面跟%y0表示将y0的值转换为字符传给之前的式子
# xy表示注释的点的坐标
# xycoords表示以哪个数据为基准
# xytext(+30,-30)表示在这个点的基础上，横坐标加30，纵坐标减30，作为注释的文字位置
# fontsize表示字体大小
# arrowprop表示箭头属性，arrowstyle="->"表示用箭头指示，
# connectionstyle='arc3,rad=0.2')表示用弧度3，曲率为0.2的弧线
# 注释方法二
# plt.text(-1,0.5, r'$This\ is\ the\ some\ text.\ \mu\ \sigma\ \alpha_t$', fontdict={'size': 16, 'color': 'g'})
# 能见度设置(防止有些图太粗，挡住坐标轴
for label in ax.get_xticklabels() + ax.get_yticklabels():  # 获取所有的刻度
    label.set_fontsize = (12)  # 改变字体大小
    label.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.7))
    # 前置色设置为白色，边框为无，透明度alpha=0.7

# 散点图
plt.figure()
n = 1024
x = np.random.normal(0, 1, n)
y = np.random.normal(0, 1, n)
col = np.arctan2(y, x)  # for color value
plt.scatter(x, y, s=75, c=col, alpha=0.5)
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.xticks(())  # 隐藏刻度
plt.yticks(())  # 隐藏刻度

# 柱状图
plt.figure()
n = 12
x = np.arange(n)
y1 = (1 - x / float(n)) * np.random.uniform(0.5, 1.0, n)  # 均匀分布
y2 = (1 - x / float(n)) * np.random.uniform(0.5, 1.0, n)
plt.xlim(-.5, n)
plt.ylim(-1.5, 1.5)
plt.xticks(())  # 隐藏刻度
plt.yticks(())  # 隐藏刻度
plt.bar(x, +y1, facecolor='#9999ff', edgecolor='white')  # 前两个参数为x和y的数据，facecolor='#9999ff'为柱状图的颜色
plt.bar(x, -y2, facecolor='#ff9999', edgecolor='white')
for a, b in zip(x, y1):  # 将x,y1的值分别传入x,y
    plt.text(a, b + 0.05, '%0.2f' % b, ha='center', va='bottom')
for i, j in zip(x, y2):  # 将x,y2的值分别传入x,y
    plt.text(i, -j - 0.05, '%0.2f' % j, ha='center', va='top')


# 等高线图
def f(x, y):
    return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)


n = 256
x = np.linspace(-3, 3, n)
y = x
plt.figure()
plt.xticks(())  # 隐藏刻度
plt.yticks(())
X, Y = np.meshgrid(x, y)  # 将x,y网格化 必要操作
plt.contourf(X, Y, f(X, Y), 15, alpha=1, cmap=plt.cm.hot)  # 前三个参数为x,y,z坐标，8表示等高线分为几层，8为10层，cmap表示色域
C = plt.contour(X, Y, f(X, Y), 15, colors='black')  # 等高线的线
plt.clabel(C, inline=True, fontsize=10)  # 给C加数字，加在线的里面，大小为10

# plot做图片
plt.figure()
a = np.array([0.31660827978, 0.36348418405, 0.423733120134,
              0.365348418405, 0.439599930621, 0.525083754405,
              0.423733120134, 0.525083754405, 0.651536351379]).reshape(3, 3)
plt.imshow(a, interpolation='nearest', cmap='bone', origin='lower')  # interpolation模糊效果，origin='lower'3*3矩阵显示方向
plt.colorbar(shrink=0.5)

# 3D图像
fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X ** 2 + Y ** 2)
Z = np.cos(R)

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
# rstride=5,cstride=1横向和纵向的区分粒度，越大，跨度越大，推荐使用小的
ax.contourf(X, Y, Z, zdir='z', offset=-2, cmap='rainbow')  # 绘制三维图片的投影zdir为投影方向，offset表示位置
ax.set_zlim3d(-3)

# 子图的制作
# 方法一 subplot()
plt.figure()
plt.subplot(2, 2, 1)  # 图片分为两行两列，子图位置为1
plt.plot([0, 1], [0, 1])
plt.subplot(2, 2, 2)  # 图片分为两行两列，子图位置为2
plt.plot([1, 0], [1, 0])
plt.subplot(2, 2, 3)  # 图片分为两行两列，子图位置为3
plt.plot([1, 0], [0, 1])
plt.subplot(2, 2, 4)  # 图片分为两行两列，子图位置为4
plt.plot([0, 1], [1, 0])
plt.figure()
plt.subplot(2, 1, 1)  # 图片分为两行1列，子图位置为1
plt.plot([0, 1], [0, 1])
plt.subplot(2, 3, 4)  # 图片分为两行3列，子图位置为4
plt.plot([1, 0], [1, 0])
plt.subplot(2, 3, 5)  # 图片分为两行3列，子图位置为5
plt.plot([1, 0], [0, 1])
plt.subplot(2, 3, 6)  # 图片分为两行3列，子图位置为6
plt.plot([1, 0], [0, 1])
# 方法2
# subplot2grid
plt.figure()
ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3, rowspan=1)  # 分成三行三列，该子图从(0,0)开始，占3列，1行
ax1.plot([1, 2], [1, 2])
ax1.set_xlabel('x')  # 跟之前不同的地方在于，子图设置横纵轴等需要加set_xxx
ax1.set_title('ax1_title')
ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=1)  # 分成三行三列，该子图从(0,0)开始，占3列，1行
ax3 = plt.subplot2grid((3, 3), (1, 2), colspan=1, rowspan=2)  # 分成三行三列，该子图从(0,0)开始，占3列，1行
ax4 = plt.subplot2grid((3, 3), (2, 0), colspan=1, rowspan=1)  # 分成三行三列，该子图从(0,0)开始，占3列，1行
ax4 = plt.subplot2grid((3, 3), (2, 1), colspan=1, rowspan=1)  # 分成三行三列，该子图从(0,0)开始，占3列，1行
# gridspec
import matplotlib.gridspec as gridspec

plt.figure()
gs = gridspec.GridSpec(3, 3)
ax1 = plt.subplot(gs[0, :])
ax2 = plt.subplot(gs[1, :2])
ax3 = plt.subplot(gs[1:, 2])
ax4 = plt.subplot(gs[-1, 0])
ax5 = plt.subplot(gs[-1, -2])
# subplots   第三种个人感觉更简洁一点
fig, ((ax11, ax12), (ax21, ax22)) = plt.subplots(2, 2, sharex='all', sharey='all')
ax11.scatter([1, 2], [1, 2])

# 图中图
# 法一
fig = plt.figure()
x = [1, 2, 3, 4, 5, 6, 7]
y = [1, 3, 4, 2, 5, 8, 6]
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8          # 这里表示图中图在的原图中的比例
ax1 = fig.add_axes([left, bottom, width, height])
ax1.plot(x, y, 'r')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('title')
left, bottom, width, height = 0.2, 0.6, 0.3, 0.25
ax2 = fig.add_axes([left, bottom, width, height])
ax2.plot(y, x, 'b')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('inside1')
plt.axes([0.6, 0.2, 0.3, 0.25])
plt.plot(y[::-1], x, 'g')
plt.xlabel('x')
plt.ylabel('y')
plt.title('inside2')

# 主次坐标轴
x = np.arange(0,10,0.1)
y1 = 0.05*x**2
y2 = -1*y1
fig,ax1 = plt.subplots()
ax2 = ax1.twinx()                # 把ax1的坐标轴做一个镜像
ax1.plot(x, y1, 'g-')
ax2.plot(x, y2, 'b--')
ax1.set_xlabel('X data')
ax1.set_ylabel('Y1', color='g')
ax2.set_ylabel('Y2', color='b')
plt.show()
