# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 23:54:39 2024

@author: adity
"""
'''
Problem: A National Zoopark in India is dealing with the problem of segregation 
of the animals based on the different attributes they have. Build a KNN model 
to automatically classify the animals. Explain any inferences you draw in the 
documentation
-------------------------------------------------------------------------------
Dataset Discription: The Zoo Animal Classification Dataset originally consists 
of 101 animals in a zoo and 16 variables with several features that describe them 
(the attributes), besides the class to which each animal belongs (the target).

That original dataset is often used in Machine Learning, as it is a complete and 
yet simple example to practice classification problems with multi-label classes. 
However, as it provides only a hundred rows, some ML algorithms might not perform well. 
That is exactly where this extended dataset fits in.
-------------------------------------------------------------------------------
Data Dictionary:

'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Zoo=pd.read_csv('C:/4-ML/Supervised Learning/Datasets/Zoo.csv')
Zoo.describe()
#Here we got five number summary

#Here we dont need first column so drop it and store other columns in new variable
zoo=Zoo.iloc[:,1:]

zoo.isnull().sum()
#There is no null values
zoo.dropna() 
zoo.columns

zoo.shape

zoo.info()


#to split train and test data
from sklearn.model_selection import train_test_split
train,test=train_test_split(zoo,test_size=0.3,random_state=0)

#KNN
from sklearn.neighbors import KNeighborsClassifier as KNC

#to find best k value
acc=[]
for i in range(3,50,2):
    neigh=KNC(n_neighbors=i)
    neigh.fit(train.iloc[:,0:16],train.iloc[:,16])
    train_acc=np.mean(neigh.predict(train.iloc[:,0:16])==train.iloc[:,16])
    test_acc=np.mean(neigh.predict(test.iloc[:,0:16])==test.iloc[:,16])
    acc.append([train_acc,test_acc])
    
plt.plot(np.arange(3,50,2),[i[0] for i in acc],'bo-')
plt.plot(np.arange(3,50,2),[i[1] for i in acc],'ro-')
plt.legend(['train','test'])

#from plots atk=5 we get best model
#model building at k=5
neigh=KNC(n_neighbors=5)
neigh.fit(train.iloc[:,0:16],train.iloc[:,16])
train_acc=np.mean(neigh.predict(train.iloc[:,0:16])==train.iloc[:,16])
test_acc=np.mean(neigh.predict(test.iloc[:,0:16])==test.iloc[:,16])
train_acc
test_acc