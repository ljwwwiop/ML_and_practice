'''
    算法：决策树 主要使用了递归，条件筛选
    理论：CS229
'''

from __future__ import print_function
import operator

from math import log
from collections import Counter

def createDataSet():
    '''
    创建数据集
    :return: 数据集 和 Label集
    '''
    dataSet =  [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
                [0, 0, 'no'],
               [0, 1, 'no']]

    labels = ['no surfacing', 'flippers']
    return dataSet,labels

def calcShannonEnt(dataSet):
    '''
    计算一下数据集中的熵，熵是指该数据的混乱程度
    :param dataSet: 数据集
    :return: 返回 每一组feature下的某个分类下，香农熵的信息期望
    '''
    print("calcShannonEnt")
    # 第一种方法，求list长度，表示计算参与训练的数据量
    numEntries = len(dataSet)
    # print type(dataSet), 'numEntries: ', numEntries
    labelCounts = {}
    for featVec in dataSet:
        # 将当前实例的标签存储，即每一行数据的最后一个数据代表的是标签
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    print(labelCounts)

    # 对于label标签的占比，求出label标签的香农熵
    shannonEnt = 0.0
    for key in labelCounts:
        # 使用所有类标签的发生频率计算类别出现的概率。
        prob = float(labelCounts[key])/numEntries
        # log base 2
        # 计算香农熵，以 2 为底求对数
        shannonEnt -= prob * log(prob, 2)
    print(shannonEnt)

    ###################
    # # 第二种方法 统计标签出现的次数
    # label_Count = Counter(data[-1] for data in dataSet)
    # probs = [p[1]/len(dataSet) for p in label_Count.items() ]
    # # 计算熵
    # shannonEnt = sum([-p*(log(p,2) for p in probs)])
    return shannonEnt

def splitDataSet(dataSet, index, value):
    '''
    按照特征划分数据集
    :param dataSet: 数据集                 待划分的数据集
    :param index: 表示每一行的index列        划分数据集的特征
    :param value: 表示index列对应的value值   需要返回的特征的值
    :return: index列为value的数据集【该数据集需要排除index列】
    '''
    print("splitDataSet")
    # 第一种方法
    retDataSet = []
    for featVec in dataSet:
        # index列的值是否为value
        if featVec[index] == value:
            # [:index]表示前index行，即若 index 为2，就是取 featVec 的前 index 行
            reducedFeatVec = featVec[:index]
            # extend 扩容只是数据本身，append是增加 []
            reducedFeatVec.extend(featVec[index + 1:])
            # [index+1:]表示从跳过 index 的 index+1行，取接下来的数据
            # 收集结果值 index列为value的行【该行需要排除index列】
            retDataSet.append(reducedFeatVec)

    # 第二种方法
    # retDataSet = [data for data in dataSet for i,v in enumerate(data) if i== index and v == value]
    print(retDataSet)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    '''
    选择最好的特征
    :param dataSet:  数据集
    :return:  最优的特征列
    '''
    print("chooseBestFeatureToSplit")
    # 第一种方法
    # 求第一行有多少列的 Feature, 最后一列是label列嘛
    numFeatures = len(dataSet[0]) - 1
    # label 的熵
    baseEntropy = calcShannonEnt(dataSet)
    # 最优的信息增益值，和最优的feature编号
    bestInfoGain,bestFeature = 0.0,-1
    for i in range(numFeatures):
        featList = [x[i] for x in dataSet]
        # 获取剔重后的集合，使用set对list数据进行去重
        uniqueVals = set(featList)
        # 一个熵的中间变量
        newEntropy = 0.0
        # 遍历某一列的value集合，计算该列的信息熵
        # 遍历当前特征中的所有唯一属性值，对每个唯一属性值划分一次数据集，计算数据集的新熵值，并对所有唯一特征值得到的熵求和。
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob*calcShannonEnt(subDataSet)
        # gain[信息增益]: 划分数据集前后的信息变化， 获取信息熵最大的值
        # 信息增益是熵的减少或者是数据无序度的减少。最后，比较所有特征中的信息增益，返回最好特征划分的索引值。
        infoGain = baseEntropy - newEntropy
        print('infoGain=', infoGain, 'bestFeature=', i, baseEntropy, newEntropy)
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    # 第二种方法
    return bestFeature

def majorityCnt(classList):
    '''
    选择出现次数最多的一个结果
    :param classList:  label列的集合
    :return: bestFeature 最优的特征列
    '''
    print("majorityCnt")
    # 第一种方法
    classCount  = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote]+=1
    # 倒叙排列classCount得到一个字典集合，然后取出第一个就是结果（yes/no），即出现次数最多的结果
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]
    # # -----------majorityCnt的第二种方式 start------------------------------------
    # major_label = Counter(classList).most_common(1)[0]
    # return major_label

def createTree(dataSet,labels):
    print("createTree")
    classList = [x[-1] for x in dataSet]
    # 如果数据集的最后一列的第一个值出现的次数=整个集合的数量，也就说只有一个类别，就只直接返回结果就行
    # 第一个停止条件: 所有的类标签完全相同，则直接返回该类标签。
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 如果数据集只有1列，那么最初出现label次数最多的一类，作为结果
    # 第二个停止条件: 使用完了所有特征，仍然不能将数据集划分成仅包含唯一类别的分组。
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    # 选择最优的列，得到最优列对应的label含义
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 获取label的名称
    bestFeatLabel = labels[bestFeat]
    #  初始化tree
    myTree = {bestFeatLabel:{}}
    # subLabels 此时就不会surface了
    subLabels = labels[:]
    del (subLabels[bestFeat])
    # 取得最有队列，然后它的branch做分类
    featValues = [e[bestFeat] for e in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        # 求出剩余的标签labels[:]
        # 遍历当前选择特征包含的所有属性值，在每个数据集划分上递归调用函数createTree()
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
        print("value",value)
    print(myTree)
    return myTree

def classify(inputTree, featLabels, testVec):
    '''
    输入节点，进行分类
    :param inputTree:  决策树模型
    :param featLabels:  标签对应的名称
    :param testVec:  测试输入的数据
    :return:  分类的结果值，需要映射label才能知道名称
    '''
    print("classify")
    # 获取tree的根节点对于的key值
    firstStr = list(inputTree.keys())[0]
    # 通过key得到根节点对应的value
    secondDict = inputTree[firstStr]
    # 判断根节点名称获取根节点在label中的先后顺序，这样就知道输入的testVec怎么开始对照树来做分类
    featIndex = featLabels.index(firstStr)
    # 测试数据，找到根节点对应的label位置，也就知道从输入的数据的第几位来开始分类
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    print('+++', firstStr, 'xxx', secondDict, '---', key, '>>>', valueOfFeat)
    # 判断分支 是否结束：判断valueOfFeat是否是dict类型
    if isinstance(valueOfFeat,dict):
        classLabel = classify(valueOfFeat,featLabels,testVec)
    else:
        classLabel = valueOfFeat
    return classLabel

def get_tree_high(tree):
    '''
    :param tree: myTree
    :return: 返回树高度
    '''
    if not isinstance(tree,dict):
        return 1

    child_trees = list(tree.values())[0].values()

    # 遍历子树，获取子树的最大高度
    max_high = 0
    for child_tree in child_trees:
        child_tree_high = get_tree_high(child_tree)
        if child_tree_high > max_high:
            max_high = child_tree_high
    return max_high +1

def fishTest():
    # 创建数据和标签
    myDat,labels = createDataSet()

    import copy
    myTree = createTree(myDat,copy.deepcopy(labels))
    print(myTree)
    # [1, 1]表示要取的分支上的节点位置，对应的结果值
    print("Is fish Yes/No? = ",classify(myTree, labels, [0, 0]))
    '''
    test:
    [0,0]  -- > NO
    [0,1]  -- > NO
    [1,0]  -- > NO
    [1,1]  -- > YES
    '''
    # 获取树高度
    # print(get_tree_high(myTree))

if __name__ == "__main__":
    fishTest()





