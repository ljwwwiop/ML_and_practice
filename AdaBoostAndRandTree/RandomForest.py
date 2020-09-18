from random import seed,randrange,random
import numpy as np
import time
'''
    随机森林代码练习
    收集数据: 提供的文本文件
    准备数据: 转换样本集
    分析数据: 手工检查数据
    训练算法: 在数据上，利用 random_forest() 函数进行优化评估，返回模型的综合分类结果
    测试算法: 在采用自定义 n_folds 份随机重抽样 进行测试评估，得出综合的预测评分
'''

# 导入CSV文件
def loadDataSet(filename):
    dataSet = []
    with open(filename,'r') as fr:
        for line in fr.readlines():
            if not line:
                continue
            lineArr = []
            for feature in line.split(','):
                # strip()返回移除字符串头尾指定的字符生成的新字符串
                str_f = feature.strip()
                # feature 有两种类型一种float,一种字母
                if str_f.isalpha():
                    # 分类标签
                    lineArr.append(str_f)
                else:
                    # 将数据集的第col列转换为float形式
                    lineArr.append(float(str_f))
            dataSet.append(lineArr)
    return dataSet

def cross_validation_split(dataset, n_folds):
    '''
    交叉验证
    :param dataset: 数据集
    :param n_folds:  将数据集划分
    :return:  将数据集进行抽重抽样 n_folds 份，数据可以重复重复抽取，每一次list的元素是无重复的
    '''
    dataset_split = list()
    # 防止直接修改了dataSet
    dataset_copy = list(dataset)
    fold_size = len(dataset)/n_folds
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            # 有放回的随机采样，有一些样本被重复采样，从而在训练集中多次出现，
            # 有的则从未在训练集中出现，此则自助采样法。从而保证每棵决策树训练集的差异性
            index = randrange(len(dataset_copy))
            # fold.append(dataset_copy.pop(index))  # 无放回的方式
            fold.append(dataset_copy[index]) # 有放回
        dataset_split.append(fold)
    # 由dataset分割出的n_folds个数据构成的列表，为了用于交叉验证
    return dataset_split

# 依据特征和特征值分割数据集
def test_split(index,value,dataSet):
    left, right = list(),list()
    for row in dataSet:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left,right

'''
Gini指数的计算问题，假如将原始数据集D切割两部分，分别为D1和D2，则
Gini(D|切割) = (|D1|/|D| ) * Gini(D1) + (|D2|/|D|) * Gini(D2)
Gini(D|切割) = Gini(D1) + Gini(D2)
'''
def gini_index(groups,class_values):
    # 个人理解: 计算代价，分类越准确，则 gini 越小
    gini = 0.0
    D = len(groups[0]) + len(groups[1])
    # class_values = [0,1]
    for class_value in class_values:
        # groups = (left,right)
        for group in groups:
            size = len(group)
            if size == 0:
                continue
            proportion = [row[-1] for row in group].count(class_value)/float(size)
            gini += float(size) / D*(proportion *(1.0 - proportion))
    return gini

# 找出分割数据集的最优特征，得到最优的特征index,特征值row[index]，以及分割完整的数据groups(left,right)
def get_split(dataset,n_features):
    class_values = list(set(row[-1] for row in dataset))
    b_index,b_value,b_score,b_group = 999,999,999,None
    features = list()
    while len(features) < n_features:
        # 把n_features特征 添加进去
        index = randrange(len(dataset[0] ) - 1)
        if index not in features:
            features.append(index)
    for index in features:
        # 在 n_features 个特征中选出最优的特征索引，并没有遍历所有特征，从而保证了每课决策树的差异性
        for row in dataset:
            groups = test_split(index,row[index],dataset)
            gini = gini_index(groups,class_values)
            # 左右两边的数量越一样，说明数据区分度不高，gini系数越大
            if gini < b_score:
                # 最后得到最优的分类特征 b_index,分类特征值 b_value,分类结果 b_groups。b_value 为分错的代价成本
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    print("b_score",b_score)
    return {'index': b_index, 'value': b_value, 'groups': b_groups}

# 输出group中出现次数较多的标签
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes),key=outcomes.count)

# 创建子分割器，递归分类，知道分类结束
def split(node, max_depth, min_size, n_features, depth):
    left,right = node['groups']
    del(node['groups'])
    # 检测是否没有切割
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # 检查最大深度
    if depth>=max_depth :
        node['left'] ,node['right'] = to_terminal(left), to_terminal(right)
        return
    # 左子树
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        # 左子树叶子节点不为1，还存在子树，则继续递归下去
        node['left'] = get_split(left, n_features)
        split(node['left'], max_depth, min_size, n_features, depth + 1)  # 递归，depth+1计算递归层数
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        # 右子树叶子节点不为1，还存在子树，则继续递归下去
        node['right'] = get_split(right, n_features)
        split(node['right'], max_depth, min_size, n_features, depth + 1)

# 创建一颗决策树
def build_tree(train, max_depth, min_size, n_features):
    '''
    :param train: 训练集
    :param max_depth:  最大深度
    :param min_size:  最少叶子节点数
    :param n_features:  选取特征个数
    :return:  root
    '''
    # 返回最优列和相关的信息
    root = get_split(train,n_features)
    # 对左右2边的数据 进行递归的调用，由于最优特征使用过，所以在后面进行使用的时候，就没有意义了
    # 例如:  性别-男女，对男使用这一特征就没任何意义了
    split(root, max_depth, min_size, n_features, 1)
    return root

# 创建数据集的随机子样本
def subsample(dataset, ratio):
    '''
    :param dataset:  训练数据集
    :param ratio:   训练数据集的样本比例
    :return:   随机抽样的训练样本
    '''
    sample = list()
    # 训练样本按比例抽样
    # round() 方法返回浮点数x的四舍五入值。
    n_sample = round(len(dataset)*ratio)
    while len(sample) < n_sample:
        # 有放回的随机采样，有一些样本被重复采样，主要保证了每棵树的差异性
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample

# 计算准确率
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct/float(len(actual)) *100.0

# 随机森林
def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
    '''
    :param train:  训练数据集
    :param test:   测试集
    :param max_depth:   决策树最大深度
    :param min_size:   叶子节点的大小
    :param sample_size:   训练数据集的样本比例
    :param n_trees:   决策树的个数
    :param n_features:   选取的特征个数
    :return:   每一行的预测结果,bagging预测最后的分类结果
    '''
    trees = list()
    for i in range(n_trees):
        # 随机抽样的训练样本，随机采样保证了决策树训练的差异性
        sample = subsample(train, sample_size)
        # 创建一颗决策树
        tree = build_tree(sample,max_depth,min_size,n_features)
        trees.append(tree)
    # 每一行预测结果,
    predictions = [bagging_predict(trees, row) for row in test]
    return predictions

# 预测模型分类结果
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'],dict):
            return predict(node['left'],row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

# 预测函数
def bagging_predict(trees, row):
    '''
    :param trees: 决策树集合
    :param row:  测试数据集的每一行数据
    :return:  返回随机森林中，决策树结果出现次数最大的
    '''
    predictions = [predict(tree,row) for tree in trees]
    print("bagging_predict predictions",predictions)
    return max(set(predictions),key=predictions.count)

# 评估算法性能，返回模型得分
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    '''
    :param dataset: 原始数据集
    :param algorithm:  使用的算法
    :param n_folds:  数据的份数
    :param args:  其他的参数
    :return:  模型得分
    '''
    # 将数据集进行抽重抽样 n_folds 份，数据可以重复重复抽取，每一次 list 的元素是无重复的
    folds = cross_validation_split(dataset,n_folds)
    scores = list()
    # 每次循环从 folds 从取出一个 fold 作为测试集，其余作为训练集，遍历整个 folds ，实现交叉验证
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        # 组合
        train_set = sum(train_set,[])
        test_set = list()
        # 目前fold是做为测试集
        for row in fold:
            row_copy = list(row)
            row_copy[-1] = None
            test_set.append(row_copy)
        predicted = algorithm(train_set,test_set,*args)
        actual = [row[-1] for row in fold]
        # 计算随机森林的预测结果的正确率
        accuracy =accuracy_metric(actual,predicted)
        scores.append(accuracy)
    return scores

if __name__ == '__main__':
    dataset = loadDataSet('sonar-all-data.txt')

    n_folds = 5
    max_depth = 20  # 决策树的深度，不能太深，不然容易过拟合
    min_size = 1    # 决策树的叶子节点最少的元素数量
    sample_size = 1.0   # 做决策树时候的样本比例
    n_features = 15     # 参数，准确性与多样性之间的权衡
    start = time.time()
    for n_trees in [1, 10, 20, 30, 40, 50]:  # 理论上树是越多越好
        scores = evaluate_algorithm(dataset,random_forest,n_folds,max_depth,min_size,sample_size,n_trees,n_features)
        # 每一次执行本文件时都能产生一个随机数
        seed(1)
        print('random=', random())
        print('Trees: %d' % n_trees)
        print('Scores: %s' % scores)
        print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
    print(u'执行时间：', time.time() - start)




