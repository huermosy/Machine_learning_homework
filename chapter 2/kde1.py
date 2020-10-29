# 导包处理
import cv2 as cv
from math import *
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D

# 图片加载函数
def image_read(filename):
    imgs = os.listdir(filename)  # 返回指定的文件夹包含的文件或文件夹的名字的列表。
    imgs.sort(key=lambda x:x[1:5])       #  按照文件名称做排序处理
    channels = len(imgs)                      #  获得指定文件中图片的个数
    data = np.empty((300, 400, 3, channels))     # 创建一个4维空数组
    for i in range(channels):
        img = cv.imread(filename + "/" + imgs[i], 1)      # 打开对应的图片,以RGB格式读入读入
        img1 = np.array(cv.imread(filename + "/" + imgs[i], 1))
        rows, cols, dims = img1.shape                  # 返回对应图像的形状，分别是高,宽,维度数
        arr = np.asarray(img)                 # 复制一个img形状的数组
        data[:, :, :, i] = arr             # 赋值
    return data, rows, cols, dims


train_data, ows, cols, dims = image_read('D:\\Learning materials\\machine_homework\\training3')    # 载入对应的训练图片,共18个训练样本
test_data, rows, cols, dims = image_read('D:\\Learning materials\\machine_homework\\test6')          # 载入对应的测试图片，有2个测试样本
test_data = np.asarray(test_data)                     # 将图片转化成数组形式

t1 = cv.getTickCount()  # 计算整个kde算法所需要的时间

h,temp = 100,1         # 带宽值设定为100，中间变量为1
P = np.zeros((rows, cols))      # 一个二维的全0数组,为最后的输出概率
P1 = np.empty((rows,cols,dims))      # 三维的全空数组
P2 = np.zeros((rows, cols))              # 二维的全0数组

for i in range(18):      #  共18个训练样本
    for j in range(rows):
        for k in range(cols):
            for l in range(dims):
                for z in range(2):        # 共2个测试样本
                    P1[j, k, l] = 1 - pow((test_data[j, k, l,z] - train_data[j, k, l, i])/h,2)
                    if P1[j, k, l] < 0:         # 如果概率密度函数值小于0，那么将其赋值为0
                        P1[j, k, l] = 0
                temp = temp * P1[j, k, l]
            P2[j, k] = temp
            temp = 1.0
    P = P + P2                         # 概率值叠加
P = 15/(8*pi*20*h**3) * P         # EP函数的表达式，其中N=20

fig = plt.figure()              # 新建子图对象,开始绘制KDE三维色彩图
ax = fig.gca(projection='3d')
X = np.linspace(1,rows,rows)
Y = np.linspace(1,cols,cols)
X, Y = np.meshgrid(Y, X)
ax.plot_surface(X, Y, P, cmap=plt.get_cmap('rainbow'))            # 设置对应的xyz轴的参数及颜色的映射
plt.title('KDE_Three_dimensional_color_map',fontsize='large', fontweight='bold')
plt.savefig("KDE_result_12.jpg")                     # 保存KDE三维色彩图
plt.show()

image = np.zeros((rows,cols))                             # 开始做二值化图像
for i in range(rows):
    for j in range(cols):
        if P[i,j] < 3.4 * 10 ** -7:
            image[i,j] = 255
        else:
            image[i,j] = 0

binary = Image.fromarray(image)   # 由数组转化成对应的二维图片
plt.figure()
plt.imshow(binary)
plt.savefig("Binary_12.jpg")         # 保存为jpg格式的图像
plt.show()
t2 = cv.getTickCount()
time = (t2 - t1)/cv.getTickFrequency()
print("time consume: {:.2f}秒".format(time))  # 秒


# 可以创新的地方:画出来的图片可能会缺角，引入 opencv 库中的函数做一个扩张的补充
# 拍的照片文件格式过大，可以用cv库更改文件样式