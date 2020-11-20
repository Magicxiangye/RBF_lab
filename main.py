import tensorflow as tf
import numpy as np
import os
import sys
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from rbf_Model import rbf_util

global z
model = rbf_util()
# 中间层的标志
global clusterCenter
global standardDeviation

kCluster = 250
hidden_size = 100
feature = 5
# 是否重新训练的标志
ifRestartT = False
# 是否有训练数据
fileTrainData = None
wholeData = None
RowCount = None
needTrainData = []
trainZ = None
# 是否有传入测试数据
predictData = None
# 监听命令行
argt = sys.argv[1:]

# 监听命令行中的方法
for v in argt:
    if v == "-restart":
        ifRestartT = True
    if v.startswith("-file="):
        tmpStr = v[len("-file="):]
        Data = pd.read_csv(tmpStr, dtype=np.float32, header=None)
        predictData = Data.as_matrix()
        RowCount = predictData.shape[0]
        for i in range(RowCount):
            needTrainData.append(predictData[i][0:feature])

        # clusterCenter, standardDeviation = model.getC_S(needTrainData, class_num=hidden_size)
        clusterCenter = joblib.load('cluster.pkl')
        standardDeviation = joblib.load('standard.pkl')
        print("predictRowCount:%s" % RowCount)
    if v.startswith("-dataFile="):
        tmpStr = v[len("-dataFile="):]
        fileStr = open(tmpStr).read()
        predictData = np.array(eval(fileStr))
        RowCount = predictData.shape[0]
        print("predictRowCount:%s" % RowCount)
    if v.startswith("-predict="):
        tmpStr = v[len("-predict="):]
        predictData = [np.fromstring(tmpStr, dtype=np.float32, sep=",")]
        print("predictData:%s" % predictData)
    if v.startswith("-traindata="):
        tmpStr = v[len("-traindata="):]
        fileTrainData = pd.read_csv(tmpStr, dtype=np.float32, header=None)
        wholeData = fileTrainData.as_matrix()  # 矩阵化
        RowCount = int(wholeData.size / wholeData[0].size)
        for i in range(RowCount):
            needTrainData.append(wholeData[i][0:feature])
        clusterCenter, standardDeviation = model.getC_S(needTrainData, class_num=hidden_size)
        # 保存的变量
        joblib.dump(clusterCenter, 'cluster.pkl')
        joblib.dump(standardDeviation, 'standard.pkl')
        print("导入训练数据开始训练")

trainResultPath = "./save/RBF_save"
random.seed()


# 先设计网络模型中的占位数
x = tf.placeholder(tf.float32, [None, feature])
yTrain = tf.placeholder(dtype=tf.float32)





# keep_prob = tf.placeholder(tf.float32)
# x = tf.nn.dropout(x,keep_prob)
# 隐藏层
# 第一层
# k-mean聚类算法获取，聚类中心，标准差（在开始训练的循环里得出结果）
# 高斯定义径向基层
z = model.kernel(x, hidden_size, s=standardDeviation, c=clusterCenter)
# 第二层的隐藏层
w = tf.Variable(tf.random_normal([hidden_size, 10]), dtype=tf.float32)
b = tf.Variable(tf.zeros([1]))
# 第二层的计算结果（过激活函数）
z1 = tf.matmul(z, w) + b
a1 = tf.nn.relu(z1)
# 第三层加输出层
w2 = tf.Variable(tf.random_normal([10, 1]), dtype=tf.float32)
b2 = tf.Variable(tf.zeros([1]))
y = tf.nn.relu(tf.matmul(a1, w2) + b2)
# 误差分析加优化（优化器也可以选其他的还有学习率）
loss = tf.reduce_mean(tf.square(y - yTrain))
#loss = tf.keras.losses.binary_crossentropy(yTrain, y)
optimizer = tf.train.AdamOptimizer(0.02)
train = optimizer.minimize(loss)
sess = tf.Session()
# 是否有已训练的模型
if ifRestartT:
    print("开始重新训练")
    sess.run(tf.global_variables_initializer())
elif os.path.exists(trainResultPath + ".index"):
    print("加载训练集：%s" % trainResultPath)
    tf.train.Saver().restore(sess, save_path=trainResultPath)
else:
    print("无保存模型，开始初始化训练")
    sess.run(tf.global_variables_initializer())

# 定一个错误率
lossSum = 0
xPlot = []
yPlot = []
predictXPlot = []
predictYPlot = []




# 判断是否是加载进训练数据
if wholeData is not None:
    for j in range(1000):
        print("第", j, "轮")
        avrLoss = 0
        for i in range(RowCount):
            xr = np.reshape(wholeData[i][0:feature], [1, feature])
            result = sess.run([train, x, y, yTrain, loss], feed_dict={x: xr, yTrain: wholeData[i][feature]})
            avrLoss = avrLoss + result[4]
            if j ==999:
                xPlot.append(result[2][0][0])
                yPlot.append(wholeData[i][feature])
        avrLoss = avrLoss / RowCount
        print(avrLoss)



    x = range(0, RowCount)

    plt.xlabel('number')
    plt.ylabel("trainResult")
    plt.plot(x, xPlot, 'o-')
    plt.plot(x, yPlot, 'o-', color='red')
    plt.grid()
    plt.show()
    isNeedSave = input("是否保存训练数据(y/n)")

    if isNeedSave == "y":
        print("正在保存训练模型")
        tf.train.Saver().save(sess, save_path=trainResultPath)
    sys.exit(0)


# 是否进测试数据
if predictData is not None:
    avgLoss = 0
    for i in range(RowCount):
        xr = np.reshape(predictData[i][0:feature], [1, feature])
        yPredict = y.eval(session=sess, feed_dict={x: xr})
        loss1 = np.sqrt((yPredict - predictData[i][feature])**2)
        avgLoss = avgLoss + loss1
        print("本次均方差为%s" % (loss1))
        #eveLoss = yPredict - predictData[i][feature]
        #print("本次均方差为%s" % (loss1))

        predictXPlot.append(yPredict[0][0])
        predictYPlot.append(predictData[i][feature])
    avgLoss = avgLoss /RowCount
    print("本次测试平均均方差为%s" % (avgLoss))
    x = range(0, RowCount)
    plt.xlabel('number')
    plt.ylabel("PredictResult")
    plt.plot(x, predictXPlot, 'o-')
    plt.plot(x, predictYPlot, 'o-', color='red')
    plt.grid()
    plt.show()
    # print(clusterCenter)
    # print(standardDeviation)

    sys.exit(0)
