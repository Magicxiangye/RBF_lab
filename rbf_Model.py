import numpy as np
import tensorflow as tf
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

PI = 3.1415926535898
min_,max_ = -5,5
global cNum
global sNum

class rbf_util:
    hidden_size = 10
    feature = 5
    #cNum = 0
    #sNum = 0
    # 使用 k-means 获取聚类中心、标准差
    def getC_S(self, x, class_num):
        # 构造聚类器
        # n_clusters: 即我们的k值；   max_iter： 最大的迭代次数
        estimator = KMeans(n_clusters=class_num, max_iter=10000)
        estimator.fit(x)  # 聚类
        # C 隐藏层函数中心
        c = estimator.cluster_centers_
        n = len(c)
        s = 0
        for i in range(n):
            j = i + 1
            while j < n:
                t = np.sum((c[i] - c[j]) ** 2)  # 取正值
                s = max(s, t)
                j = j + 1

        s = np.sqrt(s) / np.sqrt(2 * n)
        s = s.astype(np.float32)
        print(c, type(c), '\n', s, type(s))
        return c, s

    # 高斯核函数(c为中心，s为标准差)
    def kernel(self, x,hiddenNum,s,c):
        x1 = tf.tile(x, [1, hiddenNum])  # 将x水平复制 hidden次
        x2 = tf.reshape(x1, [-1, hiddenNum, self.feature])
        dist = tf.reduce_sum((x2 -c) ** 2, 2)
        return tf.exp(-dist / (2 * s ** 2))

    #def noramlization(self,data):
       #maxVals = tf.reduce_min(data)
        #ranges = maxVals - minVals
        #normData = np.zeros(np.shape(data))
        #m = data.shape[0]
       # normData = data - np.tile(minVals, (m, 1))
        #normData = normData / np.tile(ranges, (m, 1))
        #return normData