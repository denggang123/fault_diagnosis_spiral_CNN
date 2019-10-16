# -*- encoding: utf-8 -*-
"""
@Time    : 2019/8/6 10:16
@Author  : gang.deng01
"""
import os
import numpy as np

txt_dir = "./samples/"
fault_types = { "inner007.txt":0,
                "inner014.txt":1,
                "inner021.txt":2,
                "outer007.txt":3,
                "outer014.txt":4,
                "outer021.txt":5,
                "ball007.txt": 6,
                "ball014.txt": 7,
                "ball021.txt": 8,
                "normal.txt":  9
               }


class SampleCreate(object):
    """
    1.把txt文件读入矩阵，遍历矩阵各行，
    2.每一行是一个样本，按行处理矩阵元素，每行重构为一个Hankel矩阵，
    3.对Hankel矩阵进行奇异值分解，
    4.把分解的奇异值单螺旋排列为一个矩阵，
    5.以txt文件按照一定的命名格式保存,用传入参数对应的value值来实现
    """
    def __init__(self,filename):
        self.filename = filename
        # self.matrices_constructed()  # 第一次生成txt文件时调用
        print(filename)

    @staticmethod
    def interSpiralMatrix(size, matrix):
        """将特征螺旋排列为31*31的矩阵"""
        U, sigma, Vt = np.linalg.svd(matrix)
        # print(sigma)
        if (size % 2 != 1):  # size必须是奇数
            size += 1
        spiralMatrix = np.zeros([size, size], dtype=np.float32)
        x, y, side = int(size / 2), int(size / 2), size - 1
        for i in range(0, size ** 2):  # 坐标的变化是 x++ , y ++, x--, y--,,,i 表示所有的值
            spiralMatrix[y][x] = sigma[i]
            if (y <= -x + side and y <= x):  # 划分四个区域，然后就是通过直线来分开
                x += 1
            elif (-x + side < y and y < x):
                y += 1
            elif (x <= y and -x + side < y):
                x -= 1
            elif (x < y and y <= -x + side):
                y -= 1
        for matrix in spiralMatrix:
            print("\t".join(map(lambda x: str(x), matrix)))
        return spiralMatrix

    def matrices_constructed(self):
        # 1.把txt文件读入矩阵
        data = np.loadtxt(os.path.join(txt_dir,self.filename))
        sample = np.asarray(data)
        # 2.遍历矩阵各行,每行重构为一个Hankel矩阵
        h, w = sample.shape  # h表示样本总数
        row = w // 2 + 1  # Hankel矩阵的行数
        col = w // 2  # Hankel矩阵的列数
        new_sample = np.zeros([h, row, col],dtype=np.float32)  # new_sample.shape = (400,1001,1000)
        for i in range(h):
            for j in range(row):
                new_sample[i, j] = sample[i][j:col + j]
        # 3.逐个对Hankel矩阵进行SVD处理，并把singular values 螺旋排列,并保存到txt文件夹中
        for m in range(h):
            spiral_matrix = self.interSpiralMatrix(31, new_sample[m])
            if m % 4 == 3:
                np.savetxt("./processed_samples/for_test/" + str(10 * m + fault_types[self.filename]) + "_"+
                           str(fault_types[self.filename]) + ".txt", spiral_matrix)
            else:
                np.savetxt("./processed_samples/for_train/" + str(10 * m + fault_types[self.filename]) + "_" +
                           str(fault_types[self.filename]) + ".txt", spiral_matrix)

if __name__ == '__main__':
    files = os.listdir(txt_dir)
    for filename in files:
        obj = SampleCreate(filename)
