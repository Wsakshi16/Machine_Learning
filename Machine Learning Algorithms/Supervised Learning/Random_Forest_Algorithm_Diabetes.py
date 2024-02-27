# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 14:34:15 2024

@author: adity
"""

import pandas as pd

df = pd.read_csv("C:/4-ML/Supervised Learning/Datasets/diabetes(1).csv")
df.head()
df.isnull().sum()
df.describe()
df.Outcome.value_counts()
#0    500
#1    268

#there is slight imbalance in our dataset but since 
#it is not major we will not worry about it!
#train test split
X= df.drop("Outcome", axis="columns")
y=df.Outcome

from sklearn.preprocessing import StandardScaler
scaler =  StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled[:3]
#In order to make your data balanced while splitting, you can use stratify
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, random_state=10)
X_train.shape
X_test.shape
#(576, 8)
y_train.value_counts()
'''
#0    375
#1    201

'''
201/375
#0.536
y_test.value_counts()
'''
0    125
1     67
'''
67/125
#0.536
#train using stand alone model
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
#here k fold cross validation is used
scores = cross_val_score(DecisionTreeClassifier(), X, y, cv=5)
scores
#array([0.69480519, 0.64935065, 0.70779221, 0.75816993, 0.73202614])

scores.mean()
#Acuuracy= 0.7084288260758849

#Train using bagging
from sklearn.ensemble import BaggingClassifier

bag_model= BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.8,
    oob_score=True,
    random_state=0)

bag_model.fit(X_train, y_train)
bag_model.oob_score_
#0.753472222222222
#Note here we are not using test data, using OOB samples results are tested
bag_model.score(X_test, y_test)
#0.7760416666666666
#Now let us apply cross valudation
bag_model= BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.8,
    oob_score=True,
    random_state=0
    )
scores= cross_val_score(bag_model, X, y, cv=5)
scores
scores.mean()
#0.7578728461081402
#We can see some improvement in the test score with bagging classifier as comp

#Train using Random Forest
from sklearn.ensemble import RandomForestClassifier

scores=cross_val_score(RandomForestClassifier(n_estimators=50), X, y, cv=5)
scores.mean()
#0.7552839317545199





