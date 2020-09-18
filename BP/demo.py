'''
    BP 神经网络初步推导
    理论：CS229
'''
import numpy as np

def loaddataset(filename):
    '''
    :param filename: 路径名
    :return:
    '''
    fp = open(filename)
    # 数据集
    dataset = []
    # 标签集
    labelset = []
    for i in fp.readlines():
        a = i.strip().split()
        # 每个数据行的最后一个标签
        dataset.append([float(j) for j in a[:len(a)-1] ])
        labelset.append(int(float(a[-1])))
    return dataset,labelset

def parameter_initialization(x,y,z):
    '''
    :param x: 输入层神经元个数
    :param y: 隐层神经元个数
    :param z: 输出层神经元个数
    :return: 权值，和偏差值
    '''
    # 隐藏bias
    bias1 = np.random.randint(-5,5,(1,y)).astype(np.float64)
    # 输出层bias2
    bias2 = np.random.randint(-5,5,(1,z)).astype(np.float64)
    # 输入层与隐藏的连接权值
    weight1 = np.random.randint(-5,5,(x,y)).astype(np.float64)
    # 隐藏层和输出层连接权值
    weight2 = np.random.randint(-5,5,(y,z)).astype(np.float64)
    return weight1,weight2,bias1,bias2

# sigmoid 函数 或者ReLu
def sigmoid(z):
    return 1/(1+np.exp(-z))

def train(dataset,labelset,weight1,weight2,bias1,bias2):
    # 训练函数
    # x 步长
    x = 0.01
    for i in range(len(dataset)):
        # 输入数据
        inputset = np.mat(dataset[i]).astype(np.float64)
        # 数据标签
        outputset = np.mat(labelset[i]).astype(np.float64)
        # 隐藏输入
        input1 = np.dot(inputset,weight1).astype(np.float64)
        # 隐藏层输出
        output1 = sigmoid(input1 - bias1).astype(np.float64)
        # 输出层输入
        input2 = np.dot(output1,weight2).astype(np.float64)
        # 输出层输出
        output2 = sigmoid(input2 - bias2).astype(np.float64)

        # 更新公式由举证运算表示，链式法则
        a = np.multiply(output2,1-output2)
        g = np.multiply(a,outputset - output2)
        b = np.dot(g,np.transpose(weight2))
        c = np.multiply(output1,1-output1)
        e = np.multiply(b,c)

        new_bias1 = -x*e
        new_bias2 = -x*g
        new_weight1 = x*np.dot(np.transpose(inputset),e)
        new_weight2 = x*np.dot(np.transpose(output1),g)

        # 更新weight和bias
        bias1 += new_bias1
        bias2 += new_bias2
        weight1 += new_weight1
        weight2 += new_weight2
    return weight1,weight2,bias1,bias2

# test函数
def testing(dataset, labelset, weight1, weight2, bias1, bias2):
    # 记录预测正确的个数
    rightCount = 0
    # 这里可以修改，从原始数据中随机抽样，成一个测试集
    for i in range(len(dataset)):
        # 计算每一个样例通过该神经网路后的预测值
        inputset = np.mat(dataset[i]).astype(np.float64)
        outputset = np.mat(labelset[i]).astype(np.float64)
        output2 = sigmoid(np.dot(inputset,weight1) - bias1)
        output3 = sigmoid(np.dot(output2,weight2) - bias2)

        # print("w1", weight1, "w2", weight2, "b1", bias1, "b2", bias2)
        # print("output1 ",output2," ","output2 ",output3)
        # 确定其预测标签
        if output3>0.5 :
            flag = 1
        else:
            flag = 0
        if labelset[i] == flag:
            rightCount+=1
        # 预测结果
        print("预测为%d   实际为%d"%(flag, labelset[i]))
    # 返回正确率
    return rightCount/len(dataset)

if __name__ == "__main__":
    dataset,labelset = loaddataset('HorseColicTraining.txt')
    weight1,weight2,bias1,bias2 = parameter_initialization(len(dataset[0]),len(dataset[0]),1)
    # print("w1",weight1,"w2",weight2,"b1",bias1,"b2",bias2)
    # print(len(weight1),len(weight2),len(bias1),len(bias2))
    for i in range(2000):
        '''
        100 - > 正确率为0.404682
        500 - > 正确率为0.772575
        1000 - > 正确率为0.769231
        1500 - > 正确率为0.782609
        2000 - > 正确率为0.769231
        '''
        weight1, weight2, value1, value2 = train(dataset, labelset, weight1, weight2, bias1, bias2)
    rate = testing(dataset, labelset, weight1, weight2, value1, value2)
    print("正确率为%f" % (rate))




