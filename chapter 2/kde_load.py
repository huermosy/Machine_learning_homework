# 导包处理
import os, sys
import numpy as np
from PIL import Image

# 图片加载函数
def load_Img(folder_name):
    imgs = os.listdir(folder_name)
    imgs.sort(key=lambda x:x[0])
    imgNum = len(imgs)
    data = np.empty((576, 768, 3, imgNum))


    for i in range(imgNum):
        img = Image.open(folder_name + "\\" + imgs[i])
        img1 = np.array(Image.open(folder_name + "\\" + imgs[i]))
        rows, cols, dims = img1.shape
        arr = np.asarray(img)
        data[:, :, :, i] = arr
    return data, rows, cols, dims

