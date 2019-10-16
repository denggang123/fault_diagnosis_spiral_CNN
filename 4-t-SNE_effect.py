#!/usr/bin/env python
# - * - coding: utf-8 - * -
# File : 第二组实验的T-SNE图.py
# Author : 邓刚
# Date : 2019/3/26
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn import manifold, datasets


def base_setting():
    plt.xticks([0,0.10,0.20,0.30,0.40,0.5,0.6,0.7,0.8,0.9,1.0])
    plt.yticks([0,0.10,0.20,0.30,0.40,0.5,0.6,0.7,0.8,0.9,1.0])
    myfont = matplotlib.font_manager.FontProperties(fname=r'C:/Windows/Fonts/msyh.ttf') # 横纵坐标字体设置
    plt.xlabel(u'Dimension 1',fontproperties=myfont, fontweight = 'bold')  # 横坐标名称
    plt.ylabel(u'Dimension 2',fontproperties=myfont, fontweight = 'bold')  # 纵坐标名称
    plt.title('Method 3', fontsize='large',fontweight = 'bold')  # 设置字体大小与格式


# 准备数据   在test 3 里面
infile = "C:/Users/Administrator/PycharmProjects/myproject/My_cnn/paper_test/test_2/test2_last_layer10.txt"
X_data = np.loadtxt(infile)  # 数据已经准备完毕
y_data = list()
for i in range(10):
    for j in range(400):
        y_data.append(int(i)) # 标签已经准备完毕

#  t-SNE
X, y = X_data, y_data
tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
X_tsne = tsne.fit_transform(X)

print("Org data dimension is {}.Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
plt.figure(figsize=(7, 7))
plt.rcParams['savefig.dpi'] = 1600 #图片像素
plt.rcParams['figure.dpi'] = 1600 #分辨率
for i in range(X_norm.shape[0]):
    plt.text(X_norm[i, 0], X_norm[i, 1], "*", color=plt.cm.Set1(y[i]),  # "*"换成str(i)可以显示数字
             fontdict={'weight': 'bold', 'size': 8})

base_setting()
plt.show()
