'''
    KNN 是一种基本分类和回归算法
    K值选择，距离度量方式，以及分类决策规则,KNN三个基本要素
    Demo:优化约会网站的配对效果
    理论指导：
        计算新数据与样本数据集中每条数据的距离。
        对求得的所有距离进行排序（从小到大，越小表示越相似）。
        取前 k （k 一般小于等于 20 ）个样本数据对应的分类标签。
'''
from __future__ import print_function
from numpy import *
import operator
import random
from os import listdir
from collections import  Counter
import matplotlib
import matplotlib.pyplot as plt

def classify0(inX, dataSet, labels, k):
    '''
    inx [1,2,3]
    ds = [[1,2,3],[1,2,0]]
    :param inX: 分类的输入向量
    :param dataSet: 输入的训练样本集
    :param labels: 标签
    :param k: 超参数，最近邻居的数目
    :return:
    '''
    # 计算距离
    dataSetSize = dataSet.shape[0]
    # tile生成和训练样本对应的矩阵，并与训练样本求差
    # tile(inx, (3, 2)) 3 表示复制行数,2表示inx重复次数
    diffMat = tile(inX,(dataSetSize ,1)) - dataSet
    '''
        欧式距离：点到点距离
        [[1,2,3],[1,2,3]]-[[1,2,3],[1,2,0]]
        (A1-A2)^2+(B1-B2)^2+(c1-c2)^2
    '''
    sqDiffMat = diffMat**2
    # 矩阵每行相加
    sqDistance = sqDiffMat.sum(axis = 1)
    # 开方
    distances = sqDistance**0.5
    # 根据距离排序从小到大的排序，返回对应的索引位置
    # argsort() 是将x中的元素从小到大排列，提取其对应的index（索引），然后输出到y。
    # 例如: y=array([3,0,2,1,4,5]) 则，x[3]=-1最小，所以y[0]=3;x[5]=9最大，所以y[5]=5。
    sortedDistance = distances .argsort()
    # print("排序后的距离索引:",sortedDistance)

    # 选择距离最小的k个点
    classCount = {}
    for i in range(k):
        # 找到该样本的类型
        voteIlabel = labels[sortedDistance[i]]
        # print("vote",sortedDistance[i])
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    # 排序并返回出现最多的那个类型,直接max函数获取字典中value最大值
    maxClassCount = max(classCount,key = classCount.get)
    return maxClassCount

def file2matrix(filename):
    '''
    加载数据
    :param filename: 数据/文件名/路径
    :return: 数据矩阵returnMat 和 类别classLabelVector
    '''
    fp = open(filename)
    # 获取长度
    numberOfLines = len(fp.readlines())
    # 生成对应的矩阵
    # 例如: zeros(2，3)就是生成一个 2*3的矩阵，各个位置上全是 0
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []

    fp = open(filename)
    index = 0
    for line in fp.readlines():
        # str.strip([chars]) --返回移除字符串头尾指定的字符生成的新字符串
        line = line.strip()
        # 以 '\t' 切割字符串
        listFromLine = line.split('\t')
        # 每列的属性数据
        returnMat[index,:] = listFromLine[0:3]
        # 每列的类别数据，就是 label 标签数据
        classLabelVector.append(int(listFromLine[-1]))
        index+=1
    # 返回
    return returnMat,classLabelVector

def autoNorm(dataSet):
    '''
    均值归一化，消除属性之间量级不同导致的影响
    我是这样认为的，归一化就是要把你需要处理的数据经过处理后（通过某种算法）限制在你需要的一定范围内。
    首先归一化是为了后面数据处理的方便，其次是保正程序运行时收敛加快。
    y=log10(x) y=arctan(x)*2/PI　
    :param dataSet: 数据集
    :return:归一化后的数据集normDataSet,ranges和minVals即最小值与范围
    归一化公式：压缩范围(0,1)
        Y = (X-Xmin)/(Xmax-Xmin)
        min 和 max 分别是数据集中的最小特征值和最大特征值。该函数可以自动将数字特征值转化为0到1的区间。
    '''
    # 计算每种属性的最大值，最小值，范围
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    # 最大差
    ranges = maxVals - minVals
    # 第一种方法
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    # 生成与最小值之差组成的矩阵
    normDataSet = dataSet - tile(minVals,(m,1))
    # 把差值除以序列长度
    normDataSet = normDataSet/tile(ranges,(m,1))
    # 方法二
    # normDataSet = (dataSet - minVals)/ranges
    return normDataSet,ranges,minVals

def datingClassTest():
    '''
    约会网站测试
    :return:
    '''
    # 设置测试数据的一个比例（训练数据集比例 = 1- hoRatio）
    hoRatio = 0.2  # 测试范围,一部分测试一部分作为样本
    # 加载数据
    datingDataMat,datingLabels = file2matrix('./2.KNN/datingTestSet2.txt')
    # 归一化数据
    normMat,rangs,minVals = autoNorm(datingDataMat)
    # m 表示数据行数，
    m = normMat.shape[0]
    # 设置测试样本数量,numTestVecs:m 表示训练样本的数量
    numTestVecs = int(m*hoRatio)
    print("测试样本数量 = ",numTestVecs)
    errorCount = 0.0
    for i in range(numTestVecs):
        # 数据测试
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print(" the classifier came back with: %d, the real answer is: %d "%(classifierResult,datingLabels[i]))
        if(classifierResult != datingLabels[i]) :
            errorCount += 1.0
    print("the total number is :%f"%errorCount)
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))
    Show(datingDataMat,datingLabels)

# 用户个人输入预测,约会预测
def classifyPerson():
    resultList = ['基本没戏', '小概率和你约会', '很大可能可以约会']
    percentTats = float(input("平时花多少时间打游戏 ?"))
    ffMiles = float(input("一年平均旅行形成有多少 miles？"))
    iceCream = float(input("你一年大概吃多少 liters 冰淇淋？"))
    datingDataMat, datingLabels = file2matrix('./2.KNN/datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels, 3)
    print("You will probably like this person: ", resultList[classifierResult - 1])

def Show(Data,DataLabels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(Data[:, 0], Data[:, 1], 15.0 * array(DataLabels), 15.0 * array(DataLabels))
    plt.show()

# 用KNN实现一个手写数字识别系统
'''
构造一个能识别数字 0 到 9 的基于 KNN 分类器的手写数字识别系统。
划分区块
'''
def Img2Vector(filename):
    '''
    将图像数据转换为向量
    :param filename: 图片文件 32*32的
    :return: 一维矩阵,平铺拉伸 1*1024的NumPy数组
    循环读出文件的前32行，并将每行的头32个字符值存储在NumPy数组中，最后返回数组。
    '''
    returnVect = zeros((1,1024))
    fp = open(filename)
    for i in range(32):
        lineStr = fp.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    # 导入数据
    hwLabels = []
    trainingFileList = listdir('./2.KNN/trainingDigits')
    m = len(trainingFileList)
    # 每个文件是 1*1024 ,把所有文件数据压缩到一个trainingMat
    trainingMat = zeros((m,1024))
    # hwLabels存储0～9对应的index位置， trainingMat存放的每个位置对应的图片向量
    for i in range(m):
        fileNameStr = trainingFileList[i]
        # 处理打开的文件
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        # 将 32*32的矩阵->1*1024的矩阵
        trainingMat[i, :] = Img2Vector('./2.KNN/trainingDigits/%s' % fileNameStr)

    # 导入测试数据
    Alpha = 0.05
    testFileList = listdir('./2.KNN/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    _mTest = int(mTest*Alpha)
    mTest_Num = random.sample(range(mTest),_mTest)
    print(_mTest,mTest_Num)
    for i in mTest_Num:
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = Img2Vector('./2.KNN/testDigits/%s'%fileNameStr)
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print("KNN 分类后预测数值: %d ------- 实际数值: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\n 总共预测错误和 : %d" % errorCount)
    print("\n 总共预测次数：%d  总共预测错误率 : %f" % (_mTest,errorCount / float(mTest)))



if __name__ == '__main__':
    # test1()
    # datingClassTest()
    # classifyPerson()
    handwritingClassTest()






