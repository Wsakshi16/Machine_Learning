# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 15:31:16 2024

@author: adity
"""
'''
Divide the diabetes data into train and test datasets and build a Random Forest 
and Decision Tree model with Outcome as the output variable. 
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.tree import plot_tree

df=pd.read_csv("C:/4-ML/Supervised Learning/Datasets/Diabetes.csv")
df
df.columns

df
df.info()
df.isnull().sum()
'''
 Number of times pregnant        0
 Plasma glucose concentration    0
 Diastolic blood pressure        0
 Triceps skin fold thickness     0
 2-Hour serum insulin            0
 Body mass index                 0
 Diabetes pedigree function      0
 Age (years)                     0
 Class variable
 '''
 
colnames = list(df.columns)

predictors=colnames[:8]
target=colnames[8]
target

from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.3)

from sklearn.tree import DecisionTreeClassifier as DT

#help(DT)
model = DT(criterion='entropy')
model.fit(train[predictors], train[target])

from sklearn import tree
tree.plot_tree(model);

preds_test=model.predict(test[predictors])
preds_test
pd.crosstab(test[target],preds_test,rownames=["Actual"],colnames=['predictions'])
'''
predictions   NO  YES
Actual               
NO           126   32
YES           29   44
'''
np.mean(preds_test==test[target])
#0.7359307359307359

#Now let us check accuracy on training dataset
preds_train=model.predict(train[predictors])
pd.crosstab(train[target],preds_train,rownames=['Actual'],colnames=['predictions'])
'''
Actual               
NO           342    0
YES            0  195
'''
np.mean(preds_train==train[target])
#1.0
##################################################################
import pandas as pd


df=pd.read_csv("C:/4-ML/Supervised Learning/Datasets/Diabetes.csv")
df.head()
'''
    Number of times pregnant  ...   Class variable
0                          6  ...              YES
1                          1  ...               NO
2                          8  ...              YES
3                          1  ...               NO
4                          0  ...              YES

[5 rows x 9 columns]
'''

df['target'] = df.target
df[0:12]

df.rename(columns = {' Class variable':'Class'}, inplace = True) 
X=df.drop('Class', axis='columns')
y=df.Class

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=20)
#n_estimators : number of trees in the forest
model.fit(X_train, y_train)

model.score(X_test, y_test)
y_predicted=model.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
cm
'''
array([[80, 17],
       [31, 26]], dtype=int64)
'''
#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True)
plt.xlabel('predicted')
plt.ylabel('Truth')
#Text(95.72222222222221, 0.5, 'Truth')
#######################################################################################