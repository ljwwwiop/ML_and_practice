import pandas as pd
import csv
import numpy as np

def iter_search(index,All,subset):
    # All=list(All)
    distances = All
    first , second = All[index]
    if first in subset:
        pass
    else:
        subset.add(first)
        iter_search(first,distances,subset)
    if second in subset:
        pass
    else:
        subset.add(second)
        iter_search(second,distances,subset)
        pass

def max2(reader):
    top2=[]
    for row in reader:
        arr = np.array(row)
        idx = np.argsort(arr)#
        print(idx)
        idx=idx[::-1]
        top2.append(idx[1:3])
    return top2

def cluster(top2):
    set_all=set()
    for i in range(len(top2)):
        set_temp=set()
        iter_search(i,top2,set_temp)
        set_all.add(tuple(set_temp))
    return set_all
    # print(class_)

if __name__ == '__main__':
    with open('test.csv','r') as f:
        reader = csv.reader(f)
        top2=max2(reader)
        class_ = cluster(top2)
###################结果打印
        print('一共有%d个类，分别是：'%(len(class_)))
        for sub in class_:
            print(sub)