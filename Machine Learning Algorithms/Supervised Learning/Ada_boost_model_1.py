# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 09:20:07 2024

@author: adity
"""

#import necessary packages
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')
#read csv file
loan_data = pd.read_csv("C:/4-ML/Supervised Learning/Datasets/income.csv")
loan_data.columns
loan_data.head()
#let ys split the data in input and output
X = loan_data.iloc[:,0:6]
y = loan_data.iloc[:,6]
#split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#Create adaboost classifier
ada_model = AdaBoostClassifier(n_estimators=100, learning_rate=1)
#n_estimators = number of weal learners
#learning rate, it contributes weights of weak learners, by default it is 1
#Train the model
model = ada_model.fit(X_train, y_train)
#Predict the results
y_pred = model.predict(X_test)
print("accuracy",metrics.accuracy_score(y_test,y_pred))
#let us try for another base model
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
#Here base model is changed
Ada_model = AdaBoostClassifier(n_estimators=50, base_estimator = lr, learning_rate=1)
model = ada_model.fit(X_train,y_train)

y_pred = model.predict(X_test)
print("accuracy",metrics.accuracy_score(y_test,y_pred))