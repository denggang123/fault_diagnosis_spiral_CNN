# -*- encoding: utf-8 -*-
"""
@Time    : 2019/8/6 8:56
@Author  : gang.deng01
"""
import scipy
from scipy import io
from matplotlib import pyplot as plt
import numpy as np
import os

mat_dir = "../data_of_vibration_signals/"

fault_types = { "inner007":"109.mat",
                "inner014":"174.mat",
                "inner021":"213.mat",
                "outer007":"135.mat",
                "outer014":"201.mat",
                "outer021":"238.mat",
                "ball007": "122.mat",
                "ball014": "189.mat",
                "ball021": "226.mat",
                "normal":  "100.mat"
               }


class Mat2txt(object):
    """定义一个类，传入对应mat文件的key"""
    def __init__(self,key):
        self.path = os.path.join(mat_dir,fault_types[key])
        self.key = key
        self.value_of_dict = fault_types[key][0:3]
        self.data_process()  # 第一次生成txt文件时调用

    def data_process(self):
        features_struct = scipy.io.loadmat(self.path)
        if self.value_of_dict  == "174":
            x = features_struct['X173_DE_time']  # 读取数据中的有用信息,数据长度有60000以上
            # print(x[0], x[60000])
            fault = np.zeros([400, 2000], dtype=np.float32)
            for i in range(400):
                y = np.reshape(x[i * 100:(i * 100 + 2000)], [1, 2000])
                fault[i, :] = y
            np.savetxt("./samples/"+self.key+".txt", fault)
        else:
            x = features_struct['X' + self.value_of_dict + '_DE_time']  # 读取数据中的有用信息，数据长度有240000以上
            # print(x[0], x[240000])
            fault = np.zeros([400, 2000], dtype=np.float32)
            for i in range(400):
                y = np.reshape(x[i * 500:(i * 500 + 2000)], [1, 2000])
                fault[i, :] = y
            np.savetxt("./samples/" + self.key + ".txt", fault)


if __name__ == '__main__':
    for key in fault_types:
        obj = Mat2txt(key)
