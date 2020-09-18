'''
    PCA: 主要成分分析，通俗理解：就是找出一个最主要的特征，然后进行分析。
    为什么要正交？
        1 正交是为了数据有效性损失最小
        2 正交的一个原因是特征值的特征向量是正交的
    比较流行的降维技术:  独立成分分析、因子分析 和 主成分分析， 其中又以主成分分析应用最广泛。

'''
from numpy import *
import matplotlib.pyplot as plt

# 加载数据
def loadDataSet(fileName, delim='\t'):
    print(fileName)
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines() ]
    datArr = [list(map(float,line)) for line in stringArr]
    return mat(datArr)

def pca(dataMat,topNfeat=9999999):
    '''
    :param dataMat: 数据集
    :param topNfeat:   应用的N个特征
    :return:  lowDataMat 降维后的数据，reconMat 新的数据集空间
    '''
    # 计算每一列的均值
    print(dataMat)
    meanVals = mean(dataMat,axis=0)
    print("列均值",meanVals)
    # 每个向量同时都减去均值
    meanRemoved = dataMat - meanVals
    print("meanRemoved",meanRemoved)
    # cov协方差=[(x1-x均值)*(y1-y均值)+(x2-x均值)*(y2-y均值)+...+(xn-x均值)*(yn-y均值)+]/(n-1)
    '''
        方差: （一维）度量两个随机变量关系的统计量
        协方差:  （二维）度量各个维度偏离其均值的程度
        协方差矩阵: （多维）度量各个维度偏离其均值的程度
        
        当 cov(X, Y)>0时，表明X与Y正相关；(X越大，Y也越大；X越小Y，也越小。这种情况，我们称为“正相关”。)
        当 cov(X, Y)<0时，表明X与Y负相关；
        当 cov(X, Y)=0时，表明X与Y不相关。
    '''
    convMat = cov(meanRemoved,rowvar=0)
    # eigVals为特征值，eigVects为特征向量
    eigVals, eigVects = linalg.eig(mat(convMat))
    print("eigVals",eigVals)
    print("eigVects",eigVects)
    # 对特征值排序，返回从小到大的Index序号
    # x = np.array([3, 1, 2]) , np.argsort(x)
    # array([1, 2, 0])
    eigValInd = argsort(eigVals)
    eigValInd = eigValInd[:-(topNfeat):-1]
    # 重组 eigVects 最大到最小
    redEigVects = eigVects[:, eigValInd]
    # 将数据转换到新空间
    lowDDataMat = meanRemoved*redEigVects
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    print("old reconMat",reconMat)
    return lowDDataMat,reconMat

def show_picture(dataMat,reconMat):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:,0].flatten().A[0],dataMat[:,1].flatten().A[0] ,marker='^',s = 90)
    ax.scatter(reconMat[:,0].flatten().A[0] , reconMat[:,1].flatten().A[0] , marker='o', s=50, c='red')
    plt.show()

def replaceNanWithMean():
    datMat = loadDataSet('secom.data',' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        # 对value不为Nah的求均值
        # .A 返回矩阵基于的数组
        # isnan 判断是否为Nah ,~取反
        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i ])
        # 将数值为Nah赋值为均值
        datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal
    return datMat

def analyse_data(dataMat):
    meanVals = mean(dataMat,axis= 0)
    meanRemoved = dataMat - meanVals
    covMat = cov(meanRemoved,rowvar=0)
    eigVals,eigVects = linalg.eig(mat(covMat))
    print("eigVals",eigVals)
    print("特征向量",eigVects)
    eigValInd = argsort(eigVals)
    topNfeat = 20
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    print("eigValInd",eigValInd)
    cov_all_score = float(sum(eigVals))
    sum_cov_score = 0
    for i in range(0,len(eigValInd)):
        line_cov_score = float(eigVals[eigValInd[i]])
        sum_cov_score += line_cov_score
        '''
            我们发现其中有超过20%的特征值都是0。
            这就意味着这些特征都是其他特征的副本，也就是说，它们可以通过其他特征来表示，而本身并没有提供额外的信息。
            
            最前面15个值的数量级大于10^5，实际上那以后的值都变得非常小。
            这就相当于告诉我们只有部分重要特征，重要特征的数目也很快就会下降。
            
            最后，我们可能会注意到有一些小的负值，他们主要源自数值误差应该四舍五入成0.
        '''
        print('主成分: %s, 方差占比: %s%%, 累积方差占比: %s%%' % (
        format(i + 1, '2.0f'), format(line_cov_score / cov_all_score * 100, '4.2f'),
        format(sum_cov_score / cov_all_score * 100, '4.1f')))

if __name__ == '__main__':
    # dataMat = loadDataSet('testSet.txt')
    # # 只需要一个特征向量
    # lowDmat,reconMat = pca(dataMat,1)
    # # 只需要两个特征向量
    # # lowDmat,reconMat = pca(dataMat,2)
    # # show 看一看
    # show_picture(dataMat,reconMat)

    # 利用pca对半导体制造数据降维
    dataMat = replaceNanWithMean()
    print("shape ",shape(dataMat))
    # 分析数据
    analyse_data(dataMat)
    # # 只需要1一个特征向量
    lowDmat,reconMat = pca(dataMat,1)
    # show 看一看
    show_picture(dataMat,reconMat)

