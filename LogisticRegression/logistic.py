'''
    LogisticRegression(逻辑回归的一个推导)
    简单的算法案例
'''
# 案例一
from __future__ import  print_function
from numpy import *
import matplotlib.pyplot as plt

# 解析数据
def loadDataSet(file_name):
    '''
    :param file_name: 数据集文件名
    :return:
        dataMat -- 原始数据的特征
        labelMat -- 原始数据的标签，也就是每条样本对应的类别
    '''
    dataMat = []
    labelMat = []
    fr = open(file_name)
    for line in fr.readlines():
        # print(line)
        lineArr = line.strip().split()
        if len(lineArr) == 1:
            # 空元素 直接跳过
            continue
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

# sigmoid 跳跃函数
def sigmoid(inX):
    # return 1.0 / (1 + exp(-inX))
    # Tanh是Sigmoid的变形，与 sigmoid 不同的是，tanh 是0均值的。因此，实际应用中，tanh 会比 sigmoid 更好。
    return 2*1.0/(1+exp(-2*inX)) -1

# 处理函数
def gradAscent(dataMatIn,classLabels):
    '''
     Desc: 正常的梯度上身
    :param dataMatIn: 是一个2维numpy数组，每列分别代表不同的特征，每行大地表每个训练样本
    :param classLabels: 类别标签，他是一个1*100的向量，方便矩阵计算，需要将他转置，再重新赋值给LabelMat
    :return: array(weights) -- 得到的最佳回归系数
    '''
    # 转换为矩阵[[1,1,2],[1,1,2]...]
    dataMatrix = mat(dataMatIn)  # 转换为Numpy矩阵
    # transpose 矩阵转置
    labelMat = mat(classLabels).transpose()
    '''
    m - > 数据量
    n - > 矩阵的转置
    '''
    m,n = shape(dataMatrix)
    # alpha代表向目标移动步长
    alpha = 0.001
    # 迭代次数
    maxCycles = 500
    # 生成一个长度和特征值相同的矩阵，此处n为3
    # weights 代表回归习数,此处的ones((n,-1)) 创建一个长度和特征数相同的矩阵，值都是1
    weights = ones((n,1))
    for k in range(maxCycles):
        # m*3 的矩阵 * 3*1 的单位矩阵 ＝ m*1的矩阵
        # 那么乘上单位矩阵的意义，就代表: 通过公式得到的理论值
        h = sigmoid(dataMatrix*weights)
        # labelMat是实际值
        error = (labelMat - h) # 向量相减
        # 0.001* (3*m)*(m*1) 表示在每一个列上的一个误差情况，最后得出 x1,x2,xn的系数的偏移量
        weights = weights + alpha*dataMatrix.transpose()*error # 矩阵乘法，得到最后回归系数
    return array(weights)

# 后面是两种随机梯度下降方法，改善梯度下降算法带来的不便性
def stocGradAscent0(dataMatrix, classLabels):
    '''
    随机梯度下降
    :param dataMatrix: 输入数据特征
    :param classLabels:  输入数据类别标签
    :return: weights 最佳回归系数
    '''
    print(shape(dataMatrix))
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n) # 初始化长度为n的数组，元素全部为1
    for i in range(m):
        # sum(dataMatrix[i]*weights)为了求 f(x)的值， f(x)=a1*x1+b2*x2+..+nn*xn,此处求出的 h 是一个具体的数值，而不是一个矩阵
        h = sigmoid(sum(dataMatrix[i] *weights))
        # 计算真实类别与预测类别之间的差值，然后按照该差值调整回归系数
        error = classLabels[i] - h
        weights = weights + alpha*error*dataMatrix[i]
    return weights

# 随机梯度下降算法（优化,主要是最后收敛快速）
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    '''
    梯度下降，设置了迭代次数
    :param dataMatrix: 输入数据的数据特征（除去最后一列数据）
    :param classLabels: 输入数据的类别标签（最后一列数据）
    :param numIter: 迭代次数
    :return: weights得到回归系数
    '''
    m, n = shape(dataMatrix)
    weights = ones(n)  # 创建与列数相同的矩阵的系数矩阵，所有的元素都是1
    # 随机梯度, 循环150,观察是否收敛
    for j in range(numIter):
        # [0, 1, 2 .. m-1]
        dataIndex = list(range(m))
        for i in range(m):
            # i和j的不断增大，导致alpha的值不断减少，但是不为0
            alpha = 4 / (
                    1.0 + j + i
            ) + 0.0001  # alpha 会随着迭代不断减小，但永远不会减小到0，因为后边还有一个常数项0.0001
            # 随机产生一个 0～len()之间的一个值
            # random.uniform(x, y) 方法将随机生成下一个实数，它在[x,y]范围内,x是这个范围内的最小值，y是这个范围内的最大值。
            randIndex = int(random.uniform(0, len(dataIndex)))
            # sum(dataMatrix[i]*weights)为了求 f(x)的值， f(x)=a1*x1+b2*x2+..+nn*xn
            h = sigmoid(sum(dataMatrix[dataIndex[randIndex]] * weights))
            error = classLabels[dataIndex[randIndex]] - h
            # print weights, '__h=%s' % h, '__'*20, alpha, '__'*20, error, '__'*20, dataMatrix[randIndex]
            weights = weights + alpha * error * dataMatrix[dataIndex[randIndex]]
            del dataIndex[randIndex]
    return weights


# 可视化展示
def plotBestFit(dataArr, labelMat, weights):
    '''
    :param dataArr: 样本数据的特征
    :param labelMat: 样本数据的类别标签，即目标变量
    :param weights:  回归系数
    :return: None
    '''
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s = 30,c = 'red',marker = 's')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    '''
    y的由来
    首先理论上是这个样子的。
    dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
    w0*x0+w1*x1+w2*x2=f(x)
    x0最开始就设置为1叻， x2就是我们画图的y值，而f(x)被我们磨合误差给算到w0,w1,w2身上去了
    所以:  w0+w1*x+w2*y=0 => y = (-w0-w1*x)/w2   
    '''
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def simpleTest():
    # 1 收集准备数据
    dataMat,labelMat = loadDataSet("TestSet.txt")
    # 训练模型 f(x)=a1*x1+b2*x2+..+nn*xn中 (a1,b2, .., nn).T的矩阵值
    dataArr = array(dataMat)
    # weights = gradAscent(dataArr, labelMat)
    # weights = stocGradAscent0(dataArr, labelMat)
    weights = stocGradAscent1(dataArr, labelMat)
    print(weights)

    # 数据可视化
    plotBestFit(dataArr, labelMat, weights)

# --------------------------------------------------------------------------------
# 从疝气病症预测病马的死亡率
# 分类函数，根据回归系数和特征向量来计算 Sigmoid的值
def classifyVector(inX,weights):
    '''
    最终的分类函数，根据回归系数和特征向量来计算 Sigmoid 的值，大于0.5函数返回1，否则返回0
    :param inX: 特征向量，features
    :param weights: 根据梯度下降/随机梯度下降 计算得到的回归系数
    :return: 如果 prob 计算大于 0.5 函数返回 1 否则返回 0
    '''
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5: return 1.0
    else: return 0

# 打开测试数据集和训练集，并堆数据进行格式化
def colicTest():
    '''
    :return:   -- 分类错误率
    '''
    frTrain = open('HorseColicTraining.txt')
    frTest = open('HorseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    # 解析训练数据集中的数据特征和Labels
    # trainingSet 中存储训练数据集的特征，trainingLabels 存储训练数据集的样本对应的分类标签
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    # 使用 改进后的 随机梯度下降算法 求得在此数据集上的最佳回归系数 trainWeights
    trainWeights = stocGradAscent1(array(trainingSet),trainingLabels,500)
    errorCount = 0
    numTestVec = 0.0
    # 读取 测试数据集 进行测试，计算分类错误的样本条数和最终的错误率
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr),trainWeights )) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate

# 调用colicTest() 10次并求结果的平均值
def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum / float(numTests)))

if __name__ == "__main__":
    simpleTest()
    # multiTest()