import numpy as np
'''
    Numpy一些函数学习
'''
# 构建一个0-11的数组
arr = np.arange(12).reshape(3,4)
'''
    重新堆叠回一维数组,只有一行
    - ravel,reshape(-1)
    重新堆叠数组,只有一列
    - arr.reshape(-1,1)
    重新堆叠数组，只有一行
    - arr.reshape(1,-1)
    重新堆叠数组，比较关于（ravel，flatten，squeeze）
    - ravel没有返回数据源，需要变量接收
    - flatten直接返回数据源
    - squeeze只能对维数为只有一列或一行的二维数组降维
'''
b = arr.ravel()
arr2 = np.arange(4).reshape(2,2)
# print(arr.flatten())
# print(arr2)
# print(np.squeeze(arr2.reshape(1,-1)))

'''
    关于stack，vstack,hstack三个堆叠数组函数比较
    - stack堆叠数组，但是必须形状是一样的才可以堆叠,axis = 0 行堆叠，axis = 1列堆叠
    - vstack按照行的顺序把数组堆叠在一起
    - round() 方法返回浮点数x的四舍五入值。
'''
a1=[1,2,3,4]
b1=[5,6,7,8]
c1=[9,10,11,12]
arr3 = np.arange(4).reshape(-1)
c = np.stack((a1,arr3),axis=1)
d = np.vstack((a1,b1,c1))
print(d)
print(arr.reshape(2,-1))

'''
    array.T 转置
    ones() 
    - np.ones((3,1)) 生成一个3行1列 元素全部为1的数组/矩阵
    mat() 区别array 数组
    - 创建矩阵mat(zeros((3,3)))，zeros()的参数是一个tuple类型的,
      a3 = mat(a1)
      a4 = a3.tolist() 从矩阵转换到列表,
      mat(array(a3)) 从数组转换到矩阵
    zeros()
    - np.zeros((2,5)) 生成一个2行5列内容全部为0的数组
    np.multiply(),np.dot()
    - multiply（A,B）(A,B)是数组的时候，则是数组对应元素位置相乘，矩阵时候则是点积
    - dot()则是矩阵相乘，第一矩阵行*第二矩阵列,Ra秩为1时候，直接相乘并且sum后的值
'''
Arr = [[1,2]]
Arr2 = [[2,3]]
print(np.multiply(Arr,Arr2))
print(np.sum(np.multiply(np.mat(Arr),np.mat(Arr2))))
print(np.dot(np.mat(Arr),np.mat(Arr2)))
print(np.dot(Arr,Arr2))



