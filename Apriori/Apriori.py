'''
    Apriori关联算法  --  一种关于物品和物品之间的关系
    频繁项集（frequent item sets）: 经常出现在一块的物品的集合。
    关联规则（associational rules）: 暗示两种物品之间可能存在很强的关系。
    Apriori 算法的两个输入参数分别是最小支持度和数据集
'''
import numpy as np

# 加载数据集
def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

# 创建集合C1 , 即对 dataSet 进行去重，排序，放入 list 中，然后转换所有的元素为 frozenset
def createC1(dataSet):
    '''
    :param dataSet: 数据集
    :return:  返回一个frozenSet 格式的 list
    '''
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                # 遍历所有的元素，如果没有就append
                C1.append([item])
    # 数组排序
    C1.sort()
    # frozenset 表示冻结的 set 集合，元素无改变；可以把它当字典的 key 来使用
    return map(frozenset,C1)

# 计算候选数据集 CK 在数据集 D 中的支持度，并返回支持度大于最小支持度（minSupport）的数据
def scanD(D,Ck,minSupport):
    '''
    :param D:  数据集
    :param Ck:   候选集列表
    :param minSupport:  最小支持度
    :return: retList 支持度大于 minSupport集合 ,supportData 候选项集支持度数据
    '''
    # ssCnt 临时存放选数据集 Ck 的频率. 例如: a->10, b->5, c->8
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if can not in ssCnt.keys():
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float(len(list(D)))
    retList = []
    supportData = {}
    for key in ssCnt:
        # 支持度 = 候选项（key）出现的次数 / 所有数据集的数量
        support = ssCnt[key]/numItems
        if support >= minSupport:
            # 在 retList 的首位插入元素，只存储支持度满足频繁项集的值
            retList.insert(0, key)
        # 存储所有的候选项（key）和对应的支持度（support）
        supportData[key] = support
    return retList, supportData

# 输入频繁项集列表 Lk 与返回的元素个数 k，然后输出所有可能的候选项集 Ck
def aprioriGen(Lk, k):
    '''
    例如: 以 {0},{1},{2} 为输入且 k = 2 则输出 {0,1}, {0,2}, {1,2}. 以 {0,1},{0,2},{1,2} 为输入且 k = 3 则输出 {0,1,2}
    :param Lk: 频繁项集列表
    :param k:  返回的项集元素个数（若元素的前 k-2 相同，就进行合并）
    :return:  retList 元素两两合并的数据集
    '''
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1,lenLk):
            L1 = list(Lk[i])[: k-2]
            L2 = list(Lk[j])[: k-2]
            L1.sort()
            L2.sort()
            # 第一次 L1,L2 为空，元素直接进行合并，返回元素两两合并的数据集
            # if first k-2 elements are equal
            if L1 == L2:
                # set union
                # print 'union=', Lk[i] | Lk[j], Lk[i], Lk[j]
                retList.append(Lk[i] | Lk[j])
    return retList

# 找出数据集 dataSet 中支持度 >= 最小支持度的候选项集以及它们的支持度。即我们的频繁项集。
def apriori(dataSet, minSupport=0.5):
    '''
    首先构建集合 C1，然后扫描数据集来判断这些只有一个元素的项集是否满足最小支持度的要求。
    那么满足最小支持度要求的项集构成集合 L1。然后 L1 中的元素相互组合成 C2，
    C2 再进一步过滤变成 L2，然后以此类推，知道 CN 的长度为 0 时结束，即可找出所有频繁项集的支持度。
    :param dataSet:  数据集
    :param minSupport:   支持度的阈值
    :return:
        L 频繁项集
        supportData 所有元素和支持度的全集
    '''
    # C1 即对 dataSet 进行去重，排序，放入 list 中，然后转换所有的元素为 frozenset
    C1 = createC1(dataSet)
    # 对每一行进行 set 转换，然后存放到集合中
    D = map(set,dataSet)
    # 计算候选数据集 C1 在数据集 D 中的支持度，并返回支持度大于 minSupport 的数据
    L1, supportData = scanD(list(D), list(C1), minSupport)

    # L 加了一层 list, L 一共 2 层 list
    L,k = [L1],2
    # 判断 L 的第 k-2 项的数据长度是否 > 0。
    # 第一次执行时 L 为 [[frozenset([1]), frozenset([3]), frozenset([2]), frozenset([5])]]。L[k-2]=L[0]=[frozenset([1]), frozenset([3]), frozenset([2]), frozenset([5])]，最后面 k += 1
    while (len(L[k-2]) > 0):
        # 例如: 以 {0},{1},{2} 为输入且 k = 2 则输出 {0,1}, {0,2}, {1,2}. 以 {0,1},{0,2},{1,2} 为输入且 k = 3 则输出 {0,1,2}
        Ck = aprioriGen(L[k-2],k)
        # 计算候选数据集 CK 在数据集 D 中的支持度，并返回支持度大于 minSupport 的数据
        Lk,supK = scanD(D,Ck,minSupport)
        supportData.update(supK)
        if len(Lk) == 0:
            break
        # Lk 表示满足频繁子项的集合，L 元素在增加，例如:
        # l=[[set(1), set(2), set(3)]]
        # l=[[set(1), set(2), set(3)], [set(1, 2), set(2, 3)]]
        L.append(Lk)
        k += 1
    return L,supportData

def testApriori():
    # 加载测试数据集
    dataSet = loadDataSet()
    print("dataSet :",dataSet)

    # Apriori 算法生成频繁项集以及它们的支持度
    L1, supportData1 = apriori(dataSet, minSupport=0.7)
    print("L(0.7):",L1)
    print ('supportData(0.7): ', supportData1)

    # Apriori 算法生成频繁项集以及它们的支持度
    L2, supportData2 = apriori(dataSet, minSupport=0.5)
    print ('L(0.5): ', L2)
    print ('supportData(0.5): ', supportData2)

# 计算可信度（confidence）
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    '''
    :param freqSet: 频繁项集中的元素
    :param H: 频繁项集的元素集合
    :param supportData: 所有元素的支持字典
    :param brl: 关联规则列表的空数组
    :param minConf: 最小置信度
    :return: prunedH 记录 可信度大于阈值的集合
    '''
    prunedH = []
    for conseq in H:
        # 支持度定义: a -> b = support(a | b) / support(a)
        print("conseq",conseq)
        conf = supportData[freqSet] /supportData[freqSet - conseq]
        if conf >= minConf:
            # 只要买了 freqSet-conseq 集合，一定会买 conseq 集合（freqSet-conseq 集合和 conseq集合 是全集）
            print (freqSet-conseq, '-->', conseq, 'conf:', conf)
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

# 递归计算频繁项集的规则
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    '''
    :param freqSet: 频繁项集中的元素
    :param H: 频繁项集的元素集合
    :param supportData: 所有元素的支持度的字典
    :param brl: 关联规则列表的数组
    :param minConf: 最小置信度
    :return:
    '''
    # 在 m = 2 时, 跳出该递归。假设再递归一次，那么 H[0] = frozenset([2, 3, 5])，freqSet = frozenset([2, 3, 5]) ，没必要再计算 freqSet 与 H[0] 的关联规则了。
    m = len(H[0])
    if(len(freqSet)>(m+1)):
        # 第一次递归调用时生成 [frozenset([2, 3]), frozenset([2, 5]), frozenset([3, 5])]
        # 第二次 。。。没有第二次，递归条件判断时已经退出了
        Hmp1 = aprioriGen(H,m+1)
        # 返回可信度大于最小可信度的集合
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        print("Hmp1",Hmp1)
        print('len(Hmp1)=', len(Hmp1), 'len(freqSet)=', len(freqSet))
        # 计算可信度后，还有数据大于最小可信度的话，那么继续递归调用，否则跳出递归
        if (len(Hmp1) > 1):
            # print len(freqSet),  len(Hmp1[0]) + 1
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)

# 生成关联规则
def generateRules(L, supportData, minConf=0.7):
    '''
    :param L: 频繁项集列表
    :param supportData: 频繁项集支持度的字典
    :param minConf: 置信度
    :return: bigRuleList 可信度规则列表
    '''
    bigRuleList = []
    print("L",L)
    for i in range(1,len(L2)):
        print("L2",L2)
        # 获取频繁项集中每个组合的所有元素
        for freqSet in L2[i]:
            print("freqSet",freqSet)
            # 假设: freqSet= frozenset([1, 3]), H1=[frozenset([1]), frozenset([3])]
            # 组合总的元素并遍历子元素，并转化为 frozenset 集合，再存放到 list 列表中
            H1 = [frozenset([item]) for item in freqSet]
            # 2 个的组合，走 else, 2 个以上的组合，走 if
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList

def testGenerateRules():
    # 加载数据集
    dataSet = loadDataSet()
    print("dataSet",dataSet)
    # Apriori 算法生成频繁项集以及它们的支持度
    L1,supportData1 = apriori(dataSet,minSupport=0.5)
    print('L1(0.5)',L1)
    print("supportData1(0.5)",supportData1)
    # 生成关联规则
    rules = generateRules(L1, supportData1, minConf=0.5)
    print ('rules: ', rules)

def main():
    # Demo one 测试Apriori算法
    testApriori()

    # 生成关联规则
    # testGenerateRules()

if __name__ == '__main__':
    main()






