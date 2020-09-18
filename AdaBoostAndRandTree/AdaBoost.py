from numpy import *
import matplotlib.pyplot as plt

def stumpClassify(dataMat,dimen,threshVal,threshIneq):
    '''
    将数据集，按照feature列的value进行 二分法切分比较来赋值分类
    :param dataMat: 数据集
    :param dimen: 特征列
    :param threshVal: 特征值比较值
    :param threshIneq: ['lt', 'gt']
    :return: 结果集
    '''
    # 默认都是1
    retArray = ones((shape(dataMat)[0],1))
    if threshIneq == 'It':
        retArray[dataMat[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMat[:, dimen] > threshVal] = -1.0
    return retArray

def loadSimpData():
    '''
    测试数据
    :return:
        dataArr   feature对应的数据集
        labelArr  feature对应的分类标签
    '''
    dataArr = array([[1., 2.1], [2., 1.1], [1.3, 1.], [1., 1.], [2., 1.]])
    labelArr = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataArr, labelArr

def loadDataSet(filename):
    # 转换到二维数组
    numFeat = len(open(filename).readline().split('\t'))
    dataArr = []
    labelArr = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataArr.append(lineArr)
        labelArr.append(float(curLine[-1]))
    return dataArr,labelArr

def buildStump(dataArr,labelArr,D):
    '''
    得到决策树的模型
    :param dataArr:  数据标签集合
    :param labelArr:   分类标签集合
    :param D:   最初样本的所有特征权重集合
    :return:   bestStump最优的分类器模型，minError错误率，bestClasEst训练后的结果集
    '''
    # 转换数据
    dataMat = mat(dataArr)
    labelMat = mat(labelArr)
    m,n = dataMat.shape
    # 初始化数据
    numSteps = 10.0
    bestStump = {}
    bestClasEst = mat(zeros((m,1)))
    # 初始化的最小误差为无穷大
    minError = inf
    # 遍历所有的feature列,将列分成若干份，每一段以最左边的点作为分类节点
    for i in range(n):
        rangeMin = dataMat[:,i].min()
        rangeMax = dataMat[:,i].max()
        # 计算每一份元素的个数,类似均一化了
        StepSize = (rangeMax - rangeMin)/numSteps
        for j in range(-1,int(StepSize)+1):
            for inequal in ['lt', 'gt']:
                # 如果是-1，那么得到rangeMin-stepSize; 如果是numSteps，那么得到rangeMax
                threshVal = (rangeMin + float(j)*StepSize)
                # 对单层决策树进行简单分类，得到预测的分类值
                predictedVals = stumpClassify(dataMat,i,threshVal,inequal)
                # errArr 错误率
                errArr = mat(ones((m, 1)))
                # 正确为0，错误为1
                errArr[predictedVals[0] == labelMat.reshape(-1,1)] = 0
                # 将所有总和，每个特征的平均概率0.2，总和多少，就知道错误率多少了
                weightedError = D.T*errArr
                '''
                dim     表示feature列
                threshVal    表示树的分界值
                inequal     表示计算树左右颠倒的错误率的情况
                weightedError   表示整体结果的错误率
                bestClassEst     预测的最优结果
                '''
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    # print("bestStump",bestStump)
    return bestStump,minError,bestClasEst

def adaBoostTrainDS(dataArr, labelArr, numIt=40):
    '''
    adaBoost训练过程放大
    :param dataArr:  特征标签集合
    :param labelArr:   分类标签集合
    :param numIt:   实例数
    :return:  weakClassArr  弱分类器的集合，aggClassEst   预测的分类结果值
    '''
    weakClassArr = []
    m = shape(dataArr)[0]
    # 初始化一个中间变量
    D = mat(ones((m,1))/m)
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIt):
        # 得到决策树的模型
        bestStump, error, classEst = buildStump(dataArr, labelArr, D)
        # alpha 目的主要是计算每一个分类器实例的权重
        # 计算每个分类器的alpha权重值
        alpha = float(0.5*log((1.0 - error)/max(error,1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        # 分类正确: 乘积为1，不会影响结果，-1主要是下面求e的-alpha次方
        # 分类错误: 乘积为 -1，结果会受影响，所以也乘以 -1
        expon = multiply(-1 * alpha * mat(labelArr).T, classEst)
        D = multiply(D, exp(expon))
        D = D/D.sum()
        print("D.T",D.T)
        # 预测的分类结果值，在上一轮结果的基础上，进行加和操作
        aggClassEst += alpha * classEst
        # sign 判断正为1， 0为0， 负为-1，通过最终加和的权重值，判断符号。
        # 结果为: 错误的样本标签集合，因为是 !=,那么结果就是0 正, 1 负
        aggErros = multiply(sign(aggClassEst) != mat(labelArr).T,ones((m,1)))
        errorRate = aggErros.sum()/m
        if errorRate  == 0.0:
            break
    return weakClassArr,aggClassEst

def adaClassify(datToClass,ClassifierArr):
    dataMat = mat(datToClass)
    m = shape(dataMat)[0]
    aggClassEst = mat(zeros((m,1)))
    # 循环多个分类器
    for i in range(len(ClassifierArr)):
        # 前提：我们已经知道了最佳的分类器实例
        # 通过分类器来核酸每一次的分类结果，然后alpha*每一次的结果，得到最后的权重加和的值
        classEst = stumpClassify(dataMat,ClassifierArr[i]['dim'],ClassifierArr[i]['thresh'],ClassifierArr[i]['ineq'])
        aggClassEst += ClassifierArr[i]['alpha']*classEst
    #     print("aggClassEst",aggClassEst)
    # print("sign(aggClassEst)",sign(aggClassEst))
    return sign(aggClassEst)

def plotROC(predStrengths, classLabels):
    '''
    打印ROC曲线，并计算AUC的面积大小
    :param predStrengths:
    :param classLabels:
    :return:  predStrengths最终预测结果的权重值，classLabels原始数据的分类结果值
    '''
    ySum = 0.0
    # 对样本的进行求和
    numPosClass = sum(array(classLabels) == 1.0)
    # 正样本的概率
    yStep = 1/float(numPosClass)
    # 负样本的概率
    xStep = 1/float(len(classLabels) - numPosClass)
    # argsort函数返回的是数组值从小到大的索引值
    sortedIndices = predStrengths.argsort()
    # 测试结果集是否从小到大排列
    print('sortedIndicies=',sortedIndices,predStrengths[0,176], predStrengths.min(), predStrengths[0, 293], predStrengths.max())

    # 开始创建模板对象
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    # cursor光标值
    cur = (1.0,1.0)
    for index in sortedIndices.tolist()[0]:
        if classLabels[index] == 1.0:
            delx = 0
            dely = yStep
        else:
            delx = xStep
            dely = 0
            ySum += cur[1]
        # 画点连线 (x1,x2,y1,y2)
        print(cur[0], cur[0] - delx, cur[1], cur[1] - dely)
        ax.plot([cur[0], cur[0]-delx], [cur[1], cur[1]-dely], c='b')
        cur = (cur[0] - delx,cur[1] - dely)
    # 画对角的虚线
    ax.plot([0, 1], [0, 1], 'g--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    # 设置画图的范围区间 (x1, x2, y1, y2)
    ax.axis([0, 1, 0, 1])
    plt.show()
    ''''''
    print("the Area Under the Curve is: ", ySum*xStep)

if __name__ == '__main__':
    # 将5个点进行分类
    # dataArr,labelArr = loadSimpData()
    # print("dataArr",mat(dataArr))
    # print("labelArr",labelArr)

    # D表示最初值，对1进行均分为5份，平均每一个初始的概率都为0.2
    # D的目的是为了计算错误概率:  weightedError = D.T*errArr
    # D = mat(ones((5,1))/5)
    # # print(D)
    # bestStump, minError, bestClasEst = buildStump(dataArr, labelArr, D)
    # print(bestStump, minError, bestClasEst)

    # 分类器weakClassArr
    # 历史累计的分类结果集
    # weakClassArr, aggClassEst = adaBoostTrainDS(dataArr, labelArr, 9)
    # print("weakClassArr",weakClassArr,'\n',"aggClassEst",aggClassEst)
    '''
        分类的权重值：最大的值，为alpha的加和，最小值- 最大值
        特征的权重值：如果一个值误判的几率越小，那么D的特征权重越小
    '''
    # 测试数据的分类结果, 观测: aggClassEst分类的最终权重
    # print(adaClassify([0, 0], weakClassArr).T)
    # print(adaClassify([[5, 5], [0, 0]], weakClassArr).T)

    # # 马疝病数据集
    # 加载数据
    dataArr,labelArr = loadDataSet("horseColicTraining2.txt")
    weakClassArr,aggClassEst = adaBoostTrainDS(dataArr,labelArr,40)
    # 计算ROC下面AUC的面积大小
    plotROC(aggClassEst.T,labelArr)

    # 测试数据
    dataArrTest,labelArrTest = loadDataSet('horseColicTest2.txt')
    m = shape(dataArrTest)[0]
    predicting10 = adaClassify(dataArrTest,weakClassArr)
    errArr = mat(ones((m,1)))
    # 测试：计算总样本数，错误样本树，错误率
    print(m, errArr[predicting10 != mat(labelArrTest).T].sum(), errArr[predicting10 != mat(labelArrTest).T].sum() / m)




