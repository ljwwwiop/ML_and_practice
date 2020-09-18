'''
    CS229 关于朴素贝叶斯学习后，关于垃圾邮件，词语，语句分类一个小Demo实验

'''
from __future__ import print_function
from numpy import  *

##  屏蔽社区留言板的侮辱性言论
def loadDataSet():
    '''
    单词列表 postingList,所属类别 classVec
    :return: 数据集
    '''
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'], #[0,0,1,1,1......]
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList,classVec

def CreateVocabList(dataSet):
    '''
    :param dataSet:  数据集
    :return: 获取单词所有集合（不含重复元素的单词列表）
    '''
    vocabSet = set([])
    for document in dataSet:
        # 操作符 | 用于求两个集合的并集
        vocabSet = vocabSet | set(document) # 并集两个集合
    return list(vocabSet)

def setOfWords2Vec(vocabList,inputSet):
    '''
    :param vocabList:单词集合列表
    :param inputSet: 输入数据集
    :return: 匹配列表[0,1,0,1...]，其中 1与0 表示词汇表中的单词是否出现在输入的数据集中
    '''
    # 创建一个和词汇表等长的向量，并且初始化为0
    returnVec = [0]*len(vocabList)
    # 遍历文档中的所有单词，如果出现词汇表中的单词，则将输出的文档向量中的对应值设置为1
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word:%s is not in my Vocabulary!"%word)
    return returnVec

def trainNB0(trainMatrix, trainCategory):
    '''
    训练数据
    :param trainMatrix: 文件单词矩阵
    :param trainCategory: 文件对应的类别
    :return:
    '''
    # 文件总数
    numTrainDocs = len(trainMatrix)
    # 总单词数
    numWords = len(trainMatrix[0])
    # 侮辱性文件的出现概率
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    '''
        # 构造单词出现次数列表
        # p0Num 正常的统计
        # p1Num 侮辱的统计 
        # 避免单词列表中的任何一个单词为0，而导致最后的乘积为0，所以将每个单词的出现次数初始化为 1
    '''
    p0Num = ones(numWords)
    p1Num = ones(numWords)

    # 整个数据集单词出现总数，2.0根据样本/实际调查结果调整分母的值（2主要是避免分母为0）
    '''
        # p0Denom 正常的统计
        # p1Denom 侮辱的统计
    '''
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            # 累计辱骂词的频率
            p1Num += trainMatrix[i]
            # 对每篇文章的辱骂次数统计
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 类别1，即侮辱性文档的[log(P(F1|C1)),log(P(F2|C1)),log(P(F3|C1)),log(P(F4|C1)),log(P(F5|C1))....]列表
    # 类别0，即正常文档的[log(P(F1|C0)),log(P(F2|C0)),log(P(F3|C0)),log(P(F4|C0)),log(P(F5|C0))....]列表
    p1Vect = log(p1Num/p1Denom)
    p0Vect = log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify,p0Vec, p1Vec, pClass1):
    '''
    使用算法:
        # 将乘法转换为加法
        乘法: P(C|F1F2...Fn) = P(F1F2...Fn|C)P(C)/P(F1F2...Fn)
        加法: P(F1|C)*P(F2|C)....P(Fn|C)P(C) -> log(P(F1|C))+log(P(F2|C))+....+log(P(Fn|C))+log(P(C))
    :param vec2Classify: 待测数据[0,1,1,1,1...]，即要分类的向量
    :param p0Vec: 类别0，即正常文档的[log(P(F1|C0)),log(P(F2|C0)),log(P(F3|C0)),log(P(F4|C0)),log(P(F5|C0))....]列表
    :param p1Vec: 类别1，即侮辱性文档的[log(P(F1|C1)),log(P(F2|C1)),log(P(F3|C1)),log(P(F4|C1)),log(P(F5|C1))....]列表
    :param pClass1: 类别1，侮辱性文件的出现概率
    :return: 类别1 or 0
    '''
    # 使用 NumPy 数组来计算两个向量相乘的结果，这里的相乘是指对应元素相乘，即先将两个向量中的第一个元素相乘，然后将第2个元素相乘，以此类推。
    # 我的理解是: 这里的 vec2Classify * p1Vec 的意思就是将每个词与其对应的概率相关联起来
    # 可以理解为 1.单词在词汇表中的条件下，文件是good 类别的概率 也可以理解为 2.在整个空间下，文件既在词汇表中又是good类别的概率
    p1 = sum(vec2Classify*p1Vec)+ log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        # bad
        return 1
    else:
        # good
        return 0

def testingNB():
    '''
    :return: 测试朴素贝叶斯算法
    '''
    # 加载数据
    listOPosts, listClasses = loadDataSet()
    # 创建单词集合
    myVocabList = CreateVocabList(listOPosts)
    # 计算单词是否出现并且创建数据矩阵
    trainMat = []
    for postinDoc in listOPosts:
        # 返回m*len(myVocabList)的矩阵， 记录的都是0，1信息
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    # 开始训练数据
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    # 测试数据
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid','garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))

# ------------------------------------------------------------------------------------------
# 使用朴素贝叶斯过滤垃圾邮件，切分文本
import re
def textParse(bigString):
    '''
    接收一个大字符串并将其解析为字符串列表
    :param bigString: 大字符串
    :return: 去掉少于2个字符的字符串，并讲所有字符串转为小写，返回字符串列表
    '''
    # 使用正则表达式切分句子，其中分隔符是除单词，数字外的任意字符串
    listOfTokens = re.split(r'\\w*',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) >2]

def spamTest():
    '''
    对贝叶斯垃圾邮件分类器进行自动化处理
    :return: 对测试集中的每封邮件进行分类，若邮件分类错误，则错误数加 1，最后返回总的错误百分比。
    '''
    docList = []
    classList = []
    fullText = []
    for i in range(1,26):
        # 切分，解析数据，归类为1类别
        wordList = textParse(open('email/spam/%d.txt'%i,encoding='gbk').read())
        docList.append(wordList)
        classList.append(1)
        # 切分，解析数据，归分为0类别
        wordList = textParse(open('email/ham/%d.txt'%i,encoding='gbk').read())
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(0)
    # 创建词汇表
    vocabList =CreateVocabList(docList)
    trainingSet = list(range(50))
    testSet = []
    # 随机抽取 10 个邮件测试
    for i in range(10):
        # random.uniform(x, y) 随机生成一个范围为 x - y 的实数
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])

    trainMat = []
    trainClass = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainClass.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClass))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList,docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount +=1
    print('the errorCount is: ', errorCount)
    print('the testSet length is :', len(testSet))
    print('the error rate is :', float(errorCount)/len(testSet))

# test function
def testParseTest():
    print(textParse(open('email/ham/10.txt').read()))

if __name__ == '__main__':
    # testingNB()
    # testParseTest()
    spamTest()





