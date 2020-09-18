import pandas as pd
import numpy as np

def test(path):
    f = pd.read_csv(path,index_col=False)
    print(f)
    print(f.shape)
    print(f.index[10])
    print(f['122'][1])
    print([col for col in f])
    print(f._stat_axis.values.tolist())

def Classify(path):
    fd = pd.read_csv(path,index_col=0)
    col,row = fd.shape[0],fd.shape[1]
    Col_Name = [_ for _ in fd.index]
    # K = 2,桶排序 index 和 value 捆绑一起 根据value排序
    K = 2
    tmp = []
    Class_Set = set()
    for indexs in fd.index:
        res = []
        for i in range(row):
            dic = {}
            Key = Col_Name[i]
            dic[Key] = fd.loc[indexs][i]
            # if fd.loc[indexs][i] == 1:
            #     print(fd.loc[indexs][i])
            res.append(dic)
        # 处理res函数，进行排序
        res.sort(key = lambda x:list(x.values()))
        print(res[-K-1:])

        ans = []
        for Dic in res[-K-1:]:
            for x,y in Dic.items():
                ans.append(x)
        tmp.append(sorted(ans))
        ans.clear()

    print(len(tmp),tmp)
    # 进行set类别处理一下
    for t in tmp:
        Class_Set.add(tuple(t))
    print(len(Class_Set),Class_Set)

def Classify2(path):
    fd = pd.read_csv(path,index_col=0)
    col,row = fd.shape[0],fd.shape[1]
    Col_Name = [_ for _ in fd.index]
    Local_dic = {}
    for i in range(len(Col_Name)):
        Local_dic[Col_Name[i]] = i
    print(Local_dic)

    temp,index = [],0
    for indexs in fd.index:
        res = []
        for i in range(row):
            dic = {}
            Key = Col_Name[i]
            dic[Key] = fd.loc[indexs][i]
            res.append(dic)
        # 处理res函数，进行排序
        res.sort(key = lambda x:list(x.values()))
        res = res[::-1]
        # print(res)

        ans,k= {},0
        for key in res[:3]:
            for x,y in key.items():
                if y == 1:
                    k = x
                    ans[k] = []
                else:
                    ans[k].append(x)
        temp.append(ans)
        # index+=1
    print(len(temp),temp)
    n = len(temp)
    Set = set()
    # t = dfs(0,temp,Set,[],n,0)
    Get_Info(temp)
    # print(t)

def dfs(x,temp,Set,ans,n,j):
    '''
    需要进行搜索的list
    :return: [()],return_list 的 len
    '''
    if x == n :
        print(j)
        j+=1
        ans.append(Set)
        Set = set()
        return

    for num in temp[x]:
        if num not in Set:
            Set.add(num)
            dfs(x+1,temp,Set,ans,n,j)
            dfs(x - 1, temp, Set, ans, n,j)
    return ans

def Get_Info(_Info):
    # ans 用来存储集合
    ans = []
    for dic in _Info:
        if len(ans) == 0:
            for k,v in dic.items():
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
                        j+=1
                        print("Un_Set",Un_Set)
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
    print(ans)
    Check_Set(ans)

def Check_Set(ans):
    res = []
    for i in range(len(ans)-1):
        for j in range(i+1,len(ans)):
            if ans[i]&ans[j]:
                ans[i] = ans[i] |ans[j]
        res.append(ans[i])
    print(len(res),res)





if __name__ == '__main__':
    # test('test.csv')
    # test.csv - 2
    # test2.csv - 61
    # test3.csv - 2
    # test4.csv - 3
    # Classify('test.csv')
    Classify2('test.csv')




