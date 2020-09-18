import numpy as np
from scipy import io as spio
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

'''使用skLearn库进行KMeans聚类'''

def Kmeans():
    data = spio.loadmat("data.mat")
    X = data['X']
    # # n_clusters指定3类，拟合数据 n_clusters = K
    model = KMeans(n_clusters=3).fit(X)
    centroids = model.cluster_centers_ # 聚类中心

    # 原始图
    plt.scatter(X[:,0],X[:,1])
    plt.plot(centroids[:,0],centroids[:,1],'r^',markersize=10)  # 聚类中心
    plt.show()

if __name__ == "__main__":
    Kmeans()




