from sklearn.datasets import load_iris
from sklearn.manifold import MDS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MyMDS:
    def __init__(self,n_components):
        self.n_components = n_components

    def fit(self,data):
        m,n = data.shape
        dist = np.zeros((m,m))
        disti = np.zeros(m)
        distj = np.zeros(m)
        B = np.zeros((m,m))
        for i in range(m):
            dist[i] = np.sum(np.square(data[i] - data),axis=1).reshape(1,m)
        for i in range(m):
            disti[i] = np.mean(dist[i,:])
            distj[i] = np.mean(dist[:,i])
        distij = np.mean(dist)
        for i in range(m):
            for j in range(m):
                B[i,j] = -0.5*(dist[i,j] - disti[i] - distj[j] + distij)
        lamda,v = np.linalg.eigh(B)
        index = np.argsort(-lamda)[:self.n_components]
        diag_lamda = np.sqrt(np.diag(-np.sort(-lamda)[:self.n_components]))
        v_selected = v[:,index]
        z = v_selected.dot(diag_lamda)
        return z

    def Show(self,data,color=None):
        print(data)
        # print(iris.target)
        plt.scatter(data[:, 0], data[:, 1], c=color)
        plt.title("Using My Mds")
        plt.show()

class MyMDS2:
    def __init__(self,n_components):
        self.n_components = n_components

    def MDS(self,data):
        data = np.asarray(data)
        DSquare = data**2
        totalMean = np.mean(DSquare)
        columnMean = np.mean(DSquare,axis=0)
        rowMean = np.mean(DSquare,axis=1)
        B = np.zeros(DSquare.shape)
        for i in range(B.shape[0]):
            for j in range(B.shape[1]):
                B[i][j] = -0.5*(DSquare[i][j] - rowMean[i] - columnMean[j] + totalMean)
        # 求出特征值和特征向量
        eigVal,eigVec = np.linalg.eig(B)
        # 对特征值进行排序，得到排序索引
        eigValSorted_indices = np.argsort(eigVal)
        # 提取n_components个最大的特征向量
        topd_eigVec = eigVec[:,eigValSorted_indices[:-self.n_components-1:-1]]
        X = np.dot(topd_eigVec, np.sqrt(np.diag(eigVal[:-self.n_components-1:-1])))
        return X

    def Show(self,data,color = None):
        print(data)
        # print(iris.target) c = color
        plt.scatter(data[:, 0], data[:, 1])
        plt.title("Using My Mds2")
        plt.show()

class SkMDS:
    def __init__(self,n_components,data,color=None):
        self.n_components = n_components
        self.data = data
        self.c = color

    def use_mds(self):
        obj = MDS(self.n_components)
        obj.fit(self.data)
        print(obj.fit(self.data))
        iris_t2 = obj.fit_transform(self.data)
        plt.scatter(iris_t2[:, 0], iris_t2[:, 1], c=self.c)
        plt.title('Using sklearn MDS')
        plt.show()


if __name__ == '__main__':
    iris = load_iris()
    fd = pd.read_csv('test2.csv')
    # print(fd.values)
    # print("*"*20,iris.data)
    color1 = [ 0,0,0,0,0,1,1,1,1,1,2,2,2,2]
    color2 = [ 0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2]

    # clf1 = MyMDS(2)
    # # iris_t1 = clf1.fit(iris.data)
    # iris_t1 = clf1.fit(fd.values)
    # clf1.Show(iris_t1)
    # print("*" * 40)

    clf2 = SkMDS(2,fd.values)
    clf2.use_mds()
    print("*" * 40)


