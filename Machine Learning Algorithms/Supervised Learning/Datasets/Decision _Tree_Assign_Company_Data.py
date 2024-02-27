# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 14:02:13 2024

@author: adity
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.tree import plot_tree

data=pd.read_csv("Company_Data.csv")

data

from sklearn import preprocessing


le = preprocessing.LabelEncoder()

data["ShelveLoc"]=le.fit_transform(data["ShelveLoc"])
data["Urban"]=le.fit_transform(data["Urban"])
data["US"]=le.fit_transform(data["US"])

data
data.info()
data.isnull().sum()

data=data.assign(Sale=pd.cut(data['Sales'], 
                               bins=[ 0, 4, 9,15], 
                               labels=['Low', 'Medium', 'High']))

data.head(50)

# let's plot pair plot to visualise the attributes all at once
sns.pairplot(data=data, hue = "Sale")

# correlation matrix
sns.heatmap(data.corr())

target = pd.DataFrame.astype(data['Sale'], dtype="object")
df1 = data.copy()
df1 = df1.drop('Sale', axis =1)

# Defining the attributes
X = df1

target = target.fillna('').apply(str)
target

#label encoding
target = le.fit_transform(target)
target

y = target

# Splitting the data - 80:20 ratio
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size = 0.2, random_state =25)
print("Training split input- ", X_train.shape)
print("Testing split input- ", X_test.shape)

# Defining the decision tree algorithm
dtree=DecisionTreeClassifier()
dtree.fit(X_train,y_train)
print('Decision Tree Classifier Created')
#Decision Tree Classifier Created

# Predicting the values of test data
y_pred = dtree.predict(X_test)
#print("Classification report - \n", classification_report(y_test,y_pred))

#2 Way table to understand the correct and wrong  predictions

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,5))
sns.heatmap(data=cm,linewidths=.5, annot=True,square = True,  cmap = 'Blues')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(dtree.score(X_test, y_test))
plt.title(all_sample_title, size = 14)

#PLot the decision tree
from sklearn import tree
tree.plot_tree(dtree);

#Building Decision Tree Classifier using Entropy Criteria

model = DecisionTreeClassifier(criterion = 'entropy',max_depth=3)
model.fit(X_train,y_train)

# Predicting the values of test data
y_pred1 = model.predict(X_test)

#2 Way table to understand the correct and wrong  predictions

cm = confusion_matrix(y_test, y_pred1)
plt.figure(figsize=(5,5))
sns.heatmap(data=cm,linewidths=.5, annot=True,square = True,  cmap = 'Blues')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(model.score(X_test, y_test))
plt.title(all_sample_title, size = 14)

#PLot the decision tree
from sklearn import tree
tree.plot_tree(model);

#Decision Tree Regression Example
from sklearn.tree import  DecisionTreeRegressor
model1 = DecisionTreeRegressor()
model1.fit(X_train, y_train)

#Building Decision Tree Classifier (CART) using Gini Criteria
model_gini = DecisionTreeClassifier(criterion='gini', max_depth=3)


model_gini.fit(X_train, y_train)

#Prediction and computing the accuracy
pred=model.predict(X_test)
np.mean(pred==y_test)
