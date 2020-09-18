
from PIL import Image
from numpy import average, dot,linalg
import PIL.ImageOps
import os,shutil
from natsort import natsorted
import numpy as np
import glob
import pandas as pd
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from scipy.cluster import  hierarchy
import csv

end='*.png'
def distance(path='.\\CN\\'):
    images = []
    dis_all = []
    dic={}
    for root,dirs,files in os.walk(path):
        if root != path: 
            index=[]
            paths=natsorted(glob.glob(root+'/'+end))
            for i in range(len(paths)):
                dic[i]=int(paths[i].split('\\')[-1].split('.')[0])
                index.append(paths[i].split('\\')[-1].split('.')[0])
            dismat=[[0 for i in range(len(paths))] for j in range(len(paths))]
            img = np.array([np.array(Image.open(paths[i]).getdata()) for i in range(len(paths))])
            dist_out = 1-pairwise_distances(img, metric="cosine")
            yield dist_out,index,root
        else:
            pass

def iter_search(index,All,subset):
    # All=list(All)
    first , second = All[index]
    # if index in subset:
    #     pass
    # else:
    #     subset.add(index)

    if first in subset:
        pass
    else:
        subset.add(first)
        iter_search(first,All,subset)

    if second in subset:
        pass
    else:
        subset.add(second)
        iter_search(second,All,subset)
        pass

def max2(reader,dictionary):
    top2=[]
    index=[]
    for row in reader:
        arr = np.array(row)
        idx = np.argsort(arr)#
        idx=idx[::-1]
        index.append(idx[1:3])
        idx=[dictionary[i] for i in idx[1:3]]
        top2.append(idx)
    return top2,index

def cluster(top2):
    set_all=set()
    for i in range(len(top2)):
        set_temp=set()
        set_temp.add(i)
        # print(set_temp)
        iter_search(i,top2,set_temp)
        # print(len(set_temp))
        set_all.add(tuple(natsorted(set_temp)))
    set_all=list(tuple(set_all))
    temp = set_all[:]
    lenght = len(set_all)
    for i in range(lenght):
        for j in range(lenght):
            if set(set_all[i]) > set(set_all[j]) and set_all[j] in temp:
                temp.remove(set_all[j])
    return temp

class Classify_Num(object):

    def __init__(self):
        pass

    def Classify(self,PathOrData,index):
        '''
        主要分类函数，讲原本csv数据集分类
        :param path: 路径名
        :return: 类型数目，每个类的同组图片,[()]
        '''
        # # 读取数据，假若跳过了csv保存本地操作,则修改fd的读取就可以了
        # # 这边fd = dataframe 数据类型数据就可以了,然后直接注释这句话就OK了，其余不用修改
        # fd = pd.read_csv(PathOrData, index_col=0)
        fd = pd.DataFrame(data=PathOrData,  columns=index, index=index)
        print(fd)
        col, row = fd.shape[0], fd.shape[1]
        Col_Name = [_ for _ in fd.index]
        print("Col_Name_len",len(Col_Name))
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
        New_Temp = self.__Get_Info(temp)
        # print(len(New_Temp),New_Temp)
        # 返回数据 temp = [(),()]
        return len(New_Temp),New_Temp

    def __Get_Info(self,_Info):
        '''
        处理排序后的数据函数
        :param _Info: [{}] 一个链表，中间保存的字典,{}的 k - v 主要是  k:当前图片名 v: 有强联系的图片名
        :return: None
        '''
        ans = []
        print(_Info)
        for dic in _Info:
            if len(ans) == 0:
                for k, v in dic.items():
                    Set = set()
                    Set.add(k)
                    if v[0]:
                        Set.add(v[0])
                    if v[1]:
                        Set.add(v[1])
                    ans.append(Set)
            else:
                for k, v in dic.items():
                    j = 0
                    for Un_Set in ans:
                        if len(v) == 2:
                            if k not in Un_Set and v[0] not in Un_Set and v[1] not in Un_Set:
                                j += 1
                                # print("Un_Set", Un_Set)
                            else:
                                Un_Set.add(k)
                                Un_Set.add(v[0])
                                Un_Set.add(v[1])
                        elif len(v) == 1:
                            if k not in Un_Set and v[0] not in Un_Set :
                                j += 1
                                # print("Un_Set", Un_Set)
                            else:
                                Un_Set.add(k)
                                Un_Set.add(v[0])
                        else:
                            if k not in Un_Set :
                                j += 1
                                # print("Un_Set", Un_Set)
                            else:
                                Un_Set.add(k)
                    if j == len(ans):
                        Set = set()
                        Set.add(k)
                        if len(v) == 2:
                            Set.add(v[0])
                            Set.add(v[1])
                        elif len(v)==1:
                            Set.add(v[0])
                        else:
                            pass
                        ans.append(Set)
        # 因为会存在一个问题就是 比如当前字典，里面内容是后面的内容时候，需要交叉验证一下集合
        # 验证集合是否真正去重
        # print(ans)
        New_ans = self.__Check_Set(ans)
        return New_ans

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
            for j in range(i+1,len(ans)):
                if ans[i] & ans[j]:
                    ans[i] = ans[i] | ans[j]
                    ans[j].clear()
            res.append(ans[i])

        for i in range(len(res)):
            if len(res[i]) == 0:
                continue
            for j in range(i+1,len(res)):
                if res[i] &res[j]:
                    res[i]  = res[i]|res[j]
                    res[j].clear()
            tmp.append(res[i])
        return tmp

if __name__ == '__main__':
    for dis,index,root in distance(path=r'./CN/'):
        print(root)
        classifier = Classify_Num()
        _,result = classifier.Classify(dis,index)
        print(_,"result = ",result)

        for i,sub in enumerate(result):
            dstfile = root+'\\'+ str(i)+'\\'
            if not os.path.isfile(dstfile):
                os.makedirs(dstfile,mode=0o777)
            for index in natsorted([i for i in sub]):
                srcfile = root +'\\'+ str(index) + '.png'
                shutil.copy(srcfile,dstfile)
        print("完成")
















# if __name__ == '__main__':
#     for dis,dictionary,root in distance(path='.\\test\\'):
#         top2,index = max2(dis,dictionary)
#         # print(index)
#         class_ = cluster(index)
# ###################结果打印
#         print('一共有%d个类，分别是：'%(len(class_)))
#         print(natsorted([[dictionary[j]for j in i] for i in class_]))
#         for i,sub in enumerate(class_):
#             dstfile = root+'\\'+ str(i)+'\\'
#             if not os.path.isfile(dstfile):
#                 os.makedirs(dstfile,mode=0o777)
#             for index in natsorted([dictionary[i] for i in sub]):
#                 srcfile = root +'\\'+ str(index) + '.png'
#                 shutil.copy(srcfile,dstfile)
#         # index=[]
#         # with open('test.csv','r')as f:
#         #     reader = csv.reader(f)
#         #     for row in reader:
#         #         arr = np.array(row)
#         #         idx = np.argsort(arr)#
#         #         idx=idx[::-1]
#         #         index.append(idx[1:3])
#         #     print(index)
            
#         #     set_all=set()
#         #     for i in range(len(index)):
#         #         set_temp=set()
#         #         set_temp.add(i)
#         #         # print(set_temp)
#         #         iter_search(i,index,set_temp)
#         #         # print(set_temp)
#         #         set_all.add(tuple(natsorted(set_temp)))
#         #     # set_all=list(set(set_all))
#         #     set_all=list(tuple(set_all))
#         #     temp = set_all[:]
#         #     lenght = len(set_all)
#         #     for i in range(lenght):
#         #         for j in range(lenght):
#         #             if set(set_all[i]) > set(set_all[j]):
#         #                 temp.remove(set_all[j])
#         #     print(temp)