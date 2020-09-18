'''
    该文件 Class 主要处理的csv模式的文件数据进行分类
'''
import pandas as pd
import numpy as np

# 类对象
class Classify_Num(object):

    def __init__(self):
        pass

    def Classify(self,PathOrData):
        '''
        主要分类函数，讲原本csv数据集分类
        :param path: 路径名
        :return: 类型数目，每个类的同组图片,[()]
        '''
        # # 读取数据，假若跳过了csv保存本地操作,则修改fd的读取就可以了
        # # 这边fd = dataframe 数据类型数据就可以了,然后直接注释这句话就OK了，其余不用修改
        fd = pd.read_csv(PathOrData, index_col=0)
        # fd = pd.DataFrame(data=PathOrData, columns=index, index=index)
        print(fd)

        col, row = fd.shape[0], fd.shape[1]
        Col_Name = [_ for _ in fd.index]
        print("Col_Name_Len",len(Col_Name))
        Local_dic = {}
        for i in range(len(Col_Name)):
            Local_dic[Col_Name[i]] = i
        # print(Local_dic)

        temp, index = [], 0
        for indexs in fd.index:
            res = []
            for i in range(row):
                dic = {}
                Key = Col_Name[i]
                dic[Key] = fd.loc[indexs][i]
                res.append(dic)
            # 处理res函数，进行排序
            res.sort(key=lambda x: list(x.values()))
            res = res[::-1]
            # print(res)

            ans, k = {}, 0
            for key in res[:3]:
                for x, y in key.items():
                    if y == 1:
                        k = x
                        ans[k] = []
                    else:
                        ans[k].append(x)
            temp.append(ans)
            # index+=1
        t = self.__Get_Info(temp)
        # 返回数据 temp = [(),()]
        print("temp_size = ", len(t), t)
        return len(t),t

    def __Get_Info(self,_Info):
        '''
        处理排序后的数据函数
        :param _Info: [{}] 一个链表，中间保存的字典,{}的 k - v 主要是  k:当前图片名 v: 有强联系的图片名
        :return: None
        '''
        ans = []
        for dic in _Info:
            if len(ans) == 0:
                for k, v in dic.items():
                    Set = set()
                    Set.add(k)
                    Set.add(v[0])
                    Set.add(v[1])
                    ans.append(Set)
            else:
                for k, v in dic.items():
                    j = 0
                    for Un_Set in ans:
                        if k not in Un_Set and v[0] not in Un_Set and v[1] not in Un_Set:
                            j += 1
                            # print("Un_Set", Un_Set)
                        else:
                            Un_Set.add(k)
                            Un_Set.add(v[0])
                            Un_Set.add(v[1])
                    if j == len(ans):
                        Set = set()
                        Set.add(k)
                        Set.add(v[0])
                        Set.add(v[1])
                        ans.append(Set)
        # 因为会存在一个问题就是 比如当前字典，里面内容是后面的内容时候，需要交叉验证一下集合
        # 验证集合是否真正去重
        # print(ans)
        return self.__Check_Set(ans)

    def __Check_Set(self,ans):
        '''
        check 检测函数,主要是处理Get_Info，存在的特殊情况，例如[]是从前往后遍历，可能后面的k 对应的
            v 图片没有在前面出现过，此时无法加入前面集合，但是在后面又数据表明属于前一类，那么会存在集合重复
            因此主要是解决这个问题
        :param ans: [()] 链表，保存集合数据
        :return: None
        '''
        res,tmp = [],[]
        for i in range(len(ans)):
            if len(ans[i]) == 0:
                continue
            for j in range(i + 1, len(ans)):
                if ans[i] & ans[j]:
                    ans[i] = ans[i] | ans[j]
                    ans[j].clear()
            res.append(ans[i])
        # print(len(res), res)
        for i in range(len(res)):
            if len(res[i]) == 0:
                continue
            for j in range(i+1,len(res)):
                if res[i] &res[j]:
                    res[i]  = res[i]|res[j]
                    res[j].clear()
            tmp.append(res[i])
        return tmp


'''
    使用
    from Data_Modifiy import Classify_Num

    # # 类对象实例化
    test = Classify_Num()
    # # 调用Classify函数,另外两个是私有函数无法调用
    # # 'test.csv' 可以改为输入dataFrame类型数组数据就可以了
    test.Classify('test.csv')
'''

t = Classify_Num()
Size,Set = t.Classify('test2.csv')

x = 0
for i in range(Size):
    if len(Set[i]) !=0:
        x += len(Set[i])
print("x =",x)
