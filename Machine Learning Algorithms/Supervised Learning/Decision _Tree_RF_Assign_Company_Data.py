# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 14:02:13 2024

@author: adity
"""
'''
A cloth manufacturing company is interested to know about the different attributes 
contributing to high sales. Build a decision tree & random forest model with 
Sales as target variable (first convert it into categorical variable).
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

data=pd.read_csv("C:/4-ML/Supervised Learning/Datasets/Company_Data.csv")
data

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

data["ShelveLoc"]=le.fit_transform(data["ShelveLoc"])
data["Urban"]=le.fit_transform(data["Urban"])
data["US"]=le.fit_transform(data["US"])

data
data.info()
data.isnull().sum()
'''
RangeIndex: 400 entries, 0 to 399
Data columns (total 11 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   Sales        400 non-null    float64
 1   CompPrice    400 non-null    int64  
 2   Income       400 non-null    int64  
 3   Advertising  400 non-null    int64  
 4   Population   400 non-null    int64  
 5   Price        400 non-null    int64  
 6   ShelveLoc    400 non-null    int64  
 7   Age          400 non-null    int64  
 8   Education    400 non-null    int64  
 9   Urban        400 non-null    int64  
 10  US           400 non-null    int64  

Sales          0
CompPrice      0
Income         0
Advertising    0
Population     0
Price          0
ShelveLoc      0
Age            0
Education      0
Urban          0
US             0
dtype: int64
'''

data=data.assign(Sale=pd.cut(data['Sales'], 
                               bins=[ 0, 4, 9,15], 
                               labels=['Low', 'Medium', 'High']))

data.head(5)
'''
   Sales  CompPrice  Income  Advertising  ...  Education  Urban  US    Sale
0   9.50        138      73           11  ...         17      1   1    High
1  11.22        111      48           16  ...         10      1   1    High
2  10.06        113      35           10  ...         12      1   1    High
3   7.40        117     100            4  ...         14      1   1  Medium
4   4.15        141      64            3  ...         13      1   0  Medium

[5 rows x 12 columns]
'''
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
'''
0        High
1        High
2        High
3      Medium
4      Medium
 
395      High
396    Medium
397    Medium
398    Medium
399      High
'''

#label encoding
target = le.fit_transform(target)
target
'''
array([1, 1, 1, 3, 3, 1, 3, 1, 3, 3, 1, 1, 2, 1, 1, 3, 3, 1, 1, 3, 3, 1,
       3, 3, 1, 1, 3, 3, 2, 3, 1, 3, 3, 3, 2, 1, 3, 3, 3, 2, 2, 3, 1, 3,
       3, 3, 1, 3, 2, 1, 2, 3, 3, 3, 3, 3, 1, 2, 3, 3, 3, 3, 2, 3, 3, 3,
       3, 1, 1, 3, 1, 3, 3, 1, 3, 3, 1, 3, 3, 1, 3, 3, 1, 3, 2, 3, 3, 1,
       3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 1, 3, 3, 3, 3, 3, 3, 3, 2, 3, 2, 3,
       3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 1, 1, 3, 3, 3, 3, 3,
       1, 3, 2, 3, 3, 3, 1, 1, 3, 3, 3, 2, 1, 3, 2, 1, 3, 1, 1, 1, 3, 3,
       3, 3, 3, 1, 1, 1, 3, 2, 2, 3, 3, 2, 3, 3, 3, 1, 3, 1, 1, 3, 0, 3,
       3, 1, 1, 3, 3, 3, 3, 3, 1, 1, 3, 3, 3, 1, 3, 3, 3, 1, 3, 3, 3, 2,
       2, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 2, 3, 1, 1, 3, 3, 2, 3, 3, 1, 1,
       1, 3, 3, 2, 3, 3, 3, 3, 3, 1, 3, 3, 1, 3, 1, 3, 1, 1, 3, 2, 1, 1,
       3, 3, 3, 1, 3, 3, 3, 3, 1, 2, 3, 3, 1, 3, 3, 3, 2, 3, 3, 3, 3, 3,
       3, 3, 1, 3, 3, 3, 1, 3, 1, 1, 3, 3, 3, 3, 3, 2, 2, 1, 3, 3, 3, 3,
       3, 3, 3, 3, 1, 3, 1, 1, 1, 3, 3, 2, 1, 1, 3, 3, 3, 1, 1, 3, 3, 3,
       1, 1, 1, 3, 3, 1, 3, 3, 0, 3, 1, 3, 3, 3, 1, 1, 2, 1, 3, 3, 2, 1,
       3, 1, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 3, 1,
       1, 1, 3, 3, 2, 1, 3, 2, 3, 3, 3, 1, 1, 3, 3, 1, 1, 1, 3, 1, 3, 3,
       1, 3, 0, 3, 3, 3, 1, 2, 3, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1,
       3, 3, 3, 1])
'''

y = target

# Splitting the data - 80:20 ratio
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size = 0.2, random_state =25)
print("Training split input- ", X_train.shape)
print("Testing split input- ", X_test.shape)
'''
Training split input-  (320, 11)
Testing split input-  (80, 11)
'''

# Defining the decision tree algorithm
dtree=DecisionTreeClassifier()
dtree.fit(X_train,y_train)
print('Decision Tree Classifier Created')
#Decision Tree Classifier Created

# Predicting the values of test data
y_pred = dtree.predict(X_test)
print("Classification report - \n", classification_report(y_test,y_pred))
'''
Classification report - 
               precision    recall  f1-score   support

           0       1.00      1.00      1.00         1
           1       1.00      1.00      1.00        18
           2       1.00      1.00      1.00         8
           3       1.00      1.00      1.00        53

    accuracy                           1.00        80
   macro avg       1.00      1.00      1.00        80
weighted avg       1.00      1.00      1.00        80
'''
#2 Way table to understand the correct and wrong  predictions
cm = confusion_matrix(y_test, y_pred)
cm
'''
array([[ 1,  0,  0,  0],
       [ 0, 18,  0,  0],
       [ 0,  0,  8,  0],
       [ 0,  0,  0, 53]], dtype=int64)
'''

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
# DecisionTreeClassifier(criterion='entropy', max_depth=3)

# Predicting the values of test data
y_pred1 = model.predict(X_test)
y_pred1
'''
array([1, 2, 3, 3, 3, 3, 2, 3, 3, 3, 3, 2, 3, 3, 3, 3, 2, 3, 3, 1, 3, 3,
       3, 3, 3, 3, 3, 1, 3, 3, 1, 3, 3, 3, 3, 1, 1, 3, 1, 2, 3, 3, 1, 1,
       1, 3, 1, 0, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 1, 1, 1, 3, 3, 2,
       3, 1, 3, 3, 3, 3, 3, 3, 1, 3, 2, 3, 1, 3])
'''

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
#DecisionTreeRegressor()

#Building Decision Tree Classifier (CART) using Gini Criteria
model_gini = DecisionTreeClassifier(criterion='gini', max_depth=3)

model_gini.fit(X_train, y_train)
#DecisionTreeClassifier(max_depth=3)

#Prediction and computing the accuracy
pred=model.predict(X_test)
np.mean(pred==y_test)
#1.0
#######################################################################################

# =============================================================================
# Random Forest Algorithms
# =============================================================================

import pandas as pd

X=df1
y=target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=20)
#n_estimators : number of trees in the forest
model.fit(X_train, y_train)
#RandomForestClassifier(n_estimators=20)

model.score(X_test, y_test)
y_predicted=model.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
cm
'''
array([[ 0,  0,  1,  0],
       [ 0, 22,  0,  0],
       [ 0,  0,  8,  0],
       [ 0,  0,  0, 49]], dtype=int64)
'''

#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True)
plt.xlabel('predicted')
plt.ylabel('Truth')
#Text(95.72222222222221, 0.5, 'Truth')
#####################################################################################
