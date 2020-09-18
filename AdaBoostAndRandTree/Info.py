import numpy
'''
    集成方法：ensemble method，本质就是对其他算法进行组合的一种形式
    - 投票选举(bagging: 自举汇聚法 bootstrap aggregating): 是基于数据随机重抽样分类器构造的方法
    - 再学习(boosting): 是基于所有分类器的加权求和的方法
    场景：
    目前 bagging 方法最流行的版本是: 随机森林(random forest)（足够随机，就是综合判断）
        比如：你找对象，你会问你几个好朋友给出的建议，得到一个综合最好的结果，再去选择。
    目前 boosting 方法最流行的版本是: AdaBoost (学习法则)
        比如：你追对象，你前面有2个人都在追，他们都失败了，你向他们学习取经，然后你成功了。
    
'''

