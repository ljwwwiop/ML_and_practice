'''
    Kmeans算法 和 Knn 算法最大的区别
    前者是一个无监督学习，不需要label的算法
    其次前者k是簇的意思，后者k是邻近点的个数
        原理：
        第一步：簇分配，随机选K个点作为中心，计算到这K个点的距离，分为K个簇
        第二步：移动聚类中心：重新计算每个簇的中心，移动中心，重复以上步骤。
'''
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from scipy import io as spio
import time
import cv2 as cv

def KMeans():
    '''
    二维数据聚类过程
    :return:
    '''
    print("Kmeans....展示\n")
    data = spio.loadmat("data.mat")
    X = data['X']
    k = 3 # 总类数
    # 初始化类中心
    initial_centroids = np.array([
                                    [3,3],
                                    [6,2],
                                    [8,5]
                                ])
    max_iters = 10
    # 执行KMeans 算法
    runKMeans(X,initial_centroids,max_iters,True)

    '''图片压缩'''
    img_data = cv.imread("bird.png")
    # 压缩到 0 - 1
    img_data = img_data/255.0
    img_size = img_data.shape
    # 调增为N*3得矩阵，N是所有像素点个数
    X = img_data.reshape(img_size[0] *img_size[1],3 )

    K = 16
    max_iters = 5
    initial_centroids = kMeansInitCentroids(X,K)
    centroids,idx = runKMeans(X,initial_centroids,max_iters,False)
    print("KMeans运行结束")
    idx = findClosestCentroids(X, centroids)
    X_recovered = centroids[idx,:]
    X_recovered = X_recovered.reshape(img_size[0],img_size[1],3)

    print("绘制图片....\n")
    plt.subplot(1,2,1)
    plt.imshow(img_data)
    plt.title("original")
    plt.subplot(1,2,2)
    plt.imshow(X_recovered)
    plt.title("new")
    plt.show()
    print("-----------结束----------")

# 寻找到每条数据距离哪个类中心最近
def findClosestCentroids(X,initial_centroids):
    m = X.shape[0] # 数据条数
    K = initial_centroids.shape[0] # 类总数
    dis = np.zeros((m,K)) # 存储计算每个点分别到K个类的距离
    idx = np.zeros((m,1))
    '''计算每个点到每个类中心的距离'''
    for i in range(m):
        for j in range(K):
            dis[i,j] = np.dot((X[i,:]-initial_centroids[j,:]).reshape(1,-1),(X[i,:]-initial_centroids[j,:]).reshape(-1,1))
    '''返回dis每一行的最小值对应的列号，即为对应的类别'''
    dummy,idx = np.where(dis == np.min(dis,axis=1).reshape(-1,1))
    return idx[0:dis.shape[0]] # 截取一下

# 初始化类中心 -- 随机取K个点作为聚类中心
def kMeansInitCentroids(X,K):
    m = X.shape[0]
    m_arr = np.arange(0,m) # 生成 0 - m-1
    centroids = np.zeros((K,X.shape[1]))
    # 舍尔夫洗牌算法 打乱随机
    np.random.shuffle(m_arr)
    rand_indices = m_arr[:K]
    centroids = X[rand_indices,:]
    return centroids

# 聚类算法
def runKMeans(X,initial_centroids,max_iters,plot_process):
    m,n = X.shape
    K = initial_centroids.shape[0] # 类数目
    centroids = initial_centroids # 记录当前类中心
    previous_centroids = centroids # 保存上一次的
    idx = np.zeros((m,1))  # 每条数据属于哪个类

    for i in range(max_iters):
        print("迭代:%d "%(i+1))
        idx = findClosestCentroids(X,centroids)
        if plot_process:
            # 如果绘制图形,画聚类中心的移动过程
            plt = plotProcessKMeans(X,centroids,previous_centroids)
            previous_centroids = centroids
        centroids = computerCentroids(X,idx,K)
    if plot_process:
        # 最终显示结果
        plt.show()
    return centroids,idx

# 画图，聚类中心的移动过程
def plotProcessKMeans(X,centroids,previous_centroids):
    plt.scatter(X[:,0],X[:,1])
    # 上一次聚类中心
    plt.plot(previous_centroids[:,0],previous_centroids[:,1],'rx',markersize=10,linewidth=5.0)
    plt.plot(centroids[:,0],centroids[:,1],'rx',markersize=10,linewidth=5.0)
    for j in range(centroids.shape[0]):
        # 遍历每个类，画类中心的移动直线
        p1 = centroids[j,:]
        p2 = previous_centroids[j,:]
        plt.plot([p1[0],p2[0]],[p1[1],p2[1]],"->",linewidth=2.0)
    return plt

# 计算类中心 更新一个新的类中心
def computerCentroids(X,idx,K):
    n = X.shape[1]
    centroids = np.zeros((K,n))
    for i in range(K):
        # 索引要是一维的,axis=0为每一列，idx==i一次找出属于哪一类的，然后计算均值
        centroids[i,:] = np.mean(X[np.ravel(idx==i),:], axis=0).reshape(1,-1)
    return centroids

if __name__ == "__main__":
    KMeans()



