'''
Created on 2017年6月4日

@author: Arthur
to show data
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)
#print(df.head(100));
#赋值y
y=df.iloc[0:100,4].values
#重新赋值y
y=np.where(y == 'Iris-setosa',-1,1)
#赋值x,df的前100个元素的前两个属性
x=df.iloc[0:100,[0,2]].values
#显示散点图
plt.scatter(x[0:50,0], x[0:50,1], color='red', marker='o',label='setosa')
plt.scatter(x[50:100,0], x[50:100,1], color='blue', marker='x',label='versicolor')
#xy左边
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show();
