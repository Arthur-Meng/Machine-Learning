'''
Created on 2017年6月4日

@author: Arthur
to show errors
'''
from asyncio.tasks import sleep
from nltk.tag import perceptron
'''
Created on 2017年6月4日

@author: Arthur
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
class Perceptron(object):
    #外部初始化赋值
    def __init__(self,eta=0.01,n_iter=10):
        self.eta=eta
        self.n_iter=n_iter
    #内部初始化赋值
    def fit(self,x,y):
        self.w_=np.zeros(1+x.shape[1])
        self.errors_=[]
        for _ in range (self.n_iter):
            #errors记录update迭代次数
            errors=0;
            for xi,target in zip(x,y):
                update=self.eta*(target-self.predict(xi))
                self.w_[1:]+=update*xi
                self.w_[0]+=update
                #不等于0时候变化+1
                errors+=int(update!=0.0)
            self.errors_.append(errors)
        return self
    def net_input(self,x):
        #self.w_[0]是权重，np.dot(x,self.w_[1:])是变化的权重
        return np.dot(x,self.w_[1:])+self.w_[0]
    def predict(self,x):
        return np.where(self.net_input(x)>=0.0,1,-1)


df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)
#print(df.head(100));
y=df.iloc[0:100,4].values
y=np.where(y == 'Iris-setosa',-1,1)
x=df.iloc[0:100,[0,2]].values
'''
plt.scatter(x[0:50,0], x[0:50,1], color='red', marker='o',label='setosa')
plt.scatter(x[50:100,0], x[50:100,1], color='blue', marker='x',label='versicolor')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show();
'''
ppn=Perceptron(eta=0.1,n_iter=10)
ppn.fit(x,y)
plt.plot(range(1,len(ppn.errors_)+1),ppn.errors_,marker='o')
plt.xlabel('Epoches')
plt.ylabel('Number of misclassification')
plt.show()


