# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 16:21:40 2024

@author: adity
"""
'''
Problem Statements:
Given is the diabetes dataset. Build an ensemble model to correctly classify the  
outcome variable and improve your model prediction by using GridSearchCV. 
You must apply Bagging, Boosting, Stacking, and Voting on the dataset.  
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
diabetes = pd.read_csv('C:/4-ML/Supervised Learning/Datasets/Diabeted_Ensemble.csv')
print(diabetes.columns)
#Here we get all the columns present in the dataset

diabetes.head()
print("dimension of diabetes data: {}".format(diabetes.shape))
#dimension of diabetes data: (768, 9)

#diabetes.rename(columns={' Class variable':'Class_var'},inplace=True)
print(diabetes.groupby(' Class variable').size())
'''
NO     500
YES    268
dtype: int64
'''

# Mapping 'Yes' to 1 and 'No' to 0
diabetes[' Class variable'] = diabetes[' Class variable'].map({'YES': 1, 'NO': 0})

import seaborn as sns
sns.countplot(x=' Class variable', data=diabetes)
diabetes.info()
'''
#   Column                         Non-Null Count  Dtype  
---  ------                         --------------  -----  
 0    Number of times pregnant      768 non-null    int64  
 1    Plasma glucose concentration  768 non-null    int64  
 2    Diastolic blood pressure      768 non-null    int64  
 3    Triceps skin fold thickness   768 non-null    int64  
 4    2-Hour serum insulin          768 non-null    int64  
 5    Body mass index               768 non-null    float64
 6    Diabetes pedigree function    768 non-null    float64
 7    Age (years)                   768 non-null    int64  
 8    Class variable                768 non-null    int64
 '''
 #here we will get information like null values if any, count and Data types of each column.
 
diabetes.isnull().sum()
'''
 Number of times pregnant        0
 Plasma glucose concentration    0
 Diastolic blood pressure        0
 Triceps skin fold thickness     0
 2-Hour serum insulin            0
 Body mass index                 0
 Diabetes pedigree function      0
 Age (years)                     0
 Class variable                  0
'''
#here no any null values.

pd.set_option('display.float_format', '{:.2f}'.format)
diabetes.describe()
#here we will get five number summary
'''
        Number of times pregnant  ...   Class variable
count                     768.00  ...           768.00
mean                        3.85  ...             0.35
std                         3.37  ...             0.48
min                         0.00  ...             0.00
25%                         1.00  ...             0.00
50%                         3.00  ...             0.00
75%                         6.00  ...             1.00
max                        17.00  ...             1.00
'''

categorical_val = []
continous_val = []
for column in diabetes.columns:
    if len(diabetes[column].unique()) <= 10:
        categorical_val.append(column)
    else:
        continous_val.append(column)
        
# How many missing zeros are mising in each feature
feature_columns = [' Number of times pregnant', ' Plasma glucose concentration', ' Diastolic blood pressure', ' Triceps skin fold thickness', ' 2-Hour serum insulin', ' Body mass index', ' Diabetes pedigree function', ' Age (years)']
for column in feature_columns:
    print("============================================")
    print(f"{column} ==> Missing zeros : {len(diabetes.loc[diabetes[column] == 0])}")
'''
============================================
 Number of times pregnant ==> Missing zeros : 111
============================================
 Plasma glucose concentration ==> Missing zeros : 5
============================================
 Diastolic blood pressure ==> Missing zeros : 35
============================================
 Triceps skin fold thickness ==> Missing zeros : 227
============================================
 2-Hour serum insulin ==> Missing zeros : 374
============================================
 Body mass index ==> Missing zeros : 11
============================================
 Diabetes pedigree function ==> Missing zeros : 0
============================================
 Age (years) ==> Missing zeros : 0
 '''
#Here we can see that how many number of missing values in each column.

from sklearn.impute import SimpleImputer

fill_values = SimpleImputer(missing_values=0, strategy="mean", copy=False)
#here we are inserting mean value wherever missing value is there

diabetes[feature_columns] = fill_values.fit_transform(diabetes[feature_columns])

for column in feature_columns:
    print("============================================")
    print(f"{column} ==> Missing zeros : {len(diabetes.loc[diabetes[column] == 0])}")
    
'''
============================================
 Number of times pregnant ==> Missing zeros : 0
============================================
 Plasma glucose concentration ==> Missing zeros : 0
============================================
 Diastolic blood pressure ==> Missing zeros : 0
============================================
 Triceps skin fold thickness ==> Missing zeros : 0
============================================
 2-Hour serum insulin ==> Missing zeros : 0
============================================
 Body mass index ==> Missing zeros : 0
============================================
 Diabetes pedigree function ==> Missing zeros : 0
============================================
 Age (years) ==> Missing zeros : 0
'''
#Here we have make missing values zero by inserting mean values.
    
    
from sklearn.model_selection import train_test_split

diabetes.rename(columns={' Class variable':'Class_var'},inplace=True)

X = diabetes[feature_columns]
y = diabetes.Class_var

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=42)

rf_clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

confusion_matrix(y_test, rf_clf.predict(X_test))
#array([[123,  28],
#       [ 29,  51]], dtype=int64)
    
accuracy_score(y_test, rf_clf.predict(X_test))
#0.7532467532467533

# Evaluation on Training Data
confusion_matrix(y_train, rf_clf.predict(X_train))
#array([[349,   0],
#       [  0, 188]], dtype=int64)
    
accuracy_score(y_train, rf_clf.predict(X_train))
# 1.0


#####################################################3
# GridSearchCV

from sklearn.model_selection import GridSearchCV

rf_clf_grid = RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=42)

param_grid = {"max_features": [4, 5, 6, 7, 8, 9, 10], "min_samples_split": [2, 3, 10]}

grid_search = GridSearchCV(rf_clf_grid, param_grid, n_jobs = -1, cv = 5, scoring = 'accuracy')

grid_search.fit(X_train, y_train)

grid_search.best_params_
#{'max_features': 5, 'min_samples_split': 3}

cv_rf_clf_grid = grid_search.best_estimator_

from sklearn.metrics import accuracy_score, confusion_matrix

confusion_matrix(y_test, cv_rf_clf_grid.predict(X_test))
##array([[119,  32],
#       [ 25,  55]], dtype=int64)

accuracy_score(y_test, cv_rf_clf_grid.predict(X_test))
##0.7532467532467533

# Evaluation on Training Data
confusion_matrix(y_train, cv_rf_clf_grid.predict(X_train))
#array([[349,   0],
#       [  0, 188]], dtype=int64)
accuracy_score(y_train, cv_rf_clf_grid.predict(X_train))
#1.0

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


def evaluate(model, X_train, X_test, y_train, y_test):
    y_test_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    print("TRAINIG RESULTS: \n===============================")
    clf_report = pd.DataFrame(classification_report(y_train, y_train_pred, output_dict=True))
    print(f"CONFUSION MATRIX:\n{confusion_matrix(y_train, y_train_pred)}")
    print(f"ACCURACY SCORE:\n{accuracy_score(y_train, y_train_pred):.4f}")
    print(f"CLASSIFICATION REPORT:\n{clf_report}")

    print("TESTING RESULTS: \n===============================")
    clf_report = pd.DataFrame(classification_report(y_test, y_test_pred, output_dict=True))
    print(f"CONFUSION MATRIX:\n{confusion_matrix(y_test, y_test_pred)}")
    print(f"ACCURACY SCORE:\n{accuracy_score(y_test, y_test_pred):.4f}")
    print(f"CLASSIFICATION REPORT:\n{clf_report}")
    
 
################BAGGING#############   
    
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier()
bagging_clf = BaggingClassifier(base_estimator=tree, n_estimators=1500, random_state=42)
bagging_clf.fit(X_train, y_train)

evaluate(bagging_clf, X_train, X_test, y_train, y_test)
'''
TRAINIG RESULTS: 
===============================
CONFUSION MATRIX:
[[349   0]
 [  0 188]]
ACCURACY SCORE:
1.0000
CLASSIFICATION REPORT:
               0      1  accuracy  macro avg  weighted avg
precision   1.00   1.00      1.00       1.00          1.00
recall      1.00   1.00      1.00       1.00          1.00
f1-score    1.00   1.00      1.00       1.00          1.00
support   349.00 188.00      1.00     537.00        537.00
TESTING RESULTS: 
===============================
CONFUSION MATRIX:
[[119  32]
 [ 24  56]]
ACCURACY SCORE:
0.7576
CLASSIFICATION REPORT:
               0     1  accuracy  macro avg  weighted avg
precision   0.83  0.64      0.76       0.73          0.76
recall      0.79  0.70      0.76       0.74          0.76
f1-score    0.81  0.67      0.76       0.74          0.76
support   151.00 80.00      0.76     231.00        231.00
'''

scores = {
    'Bagging Classifier': {
        'Train': accuracy_score(y_train, bagging_clf.predict(X_train)),
        'Test': accuracy_score(y_test, bagging_clf.predict(X_test)),
    },
}
################ADA Boosting#############
from sklearn.ensemble import AdaBoostClassifier

ada_boost_clf = AdaBoostClassifier(n_estimators=30)
ada_boost_clf.fit(X_train, y_train)
#AdaBoostClassifier(n_estimators=30)

evaluate(ada_boost_clf, X_train, X_test, y_train, y_test)
'''
TRAINIG RESULTS: 
===============================
CONFUSION MATRIX:
[[310  39]
 [ 51 137]]
ACCURACY SCORE:
0.8324
CLASSIFICATION REPORT:
               0      1  accuracy  macro avg  weighted avg
precision   0.86   0.78      0.83       0.82          0.83
recall      0.89   0.73      0.83       0.81          0.83
f1-score    0.87   0.75      0.83       0.81          0.83
support   349.00 188.00      0.83     537.00        537.00
TESTING RESULTS: 
===============================
CONFUSION MATRIX:
[[123  28]
 [ 27  53]]
ACCURACY SCORE:
0.7619
CLASSIFICATION REPORT:
               0     1  accuracy  macro avg  weighted avg
precision   0.82  0.65      0.76       0.74          0.76
recall      0.81  0.66      0.76       0.74          0.76
f1-score    0.82  0.66      0.76       0.74          0.76
support   151.00 80.00      0.76     231.00        231.00

'''

scores['AdaBoost'] = {
        'Train': accuracy_score(y_train, ada_boost_clf.predict(X_train)),
        'Test': accuracy_score(y_test, ada_boost_clf.predict(X_test)),
    }

################GradientBoosting#############

from sklearn.ensemble import GradientBoostingClassifier

grad_boost_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
#GradientBoostingClassifier(n_estimators=100, random_state=42)

grad_boost_clf.fit(X_train, y_train)
#GradientBoostingClassifier(random_state=42)

evaluate(grad_boost_clf, X_train, X_test, y_train, y_test)
'''
TRAINIG RESULTS: 
===============================
CONFUSION MATRIX:
[[342   7]
 [ 19 169]]
ACCURACY SCORE:
0.9516
CLASSIFICATION REPORT:
               0      1  accuracy  macro avg  weighted avg
precision   0.95   0.96      0.95       0.95          0.95
recall      0.98   0.90      0.95       0.94          0.95
f1-score    0.96   0.93      0.95       0.95          0.95
support   349.00 188.00      0.95     537.00        537.00
TESTING RESULTS: 
===============================
CONFUSION MATRIX:
[[116  35]
 [ 26  54]]
ACCURACY SCORE:
0.7359
CLASSIFICATION REPORT:
               0     1  accuracy  macro avg  weighted avg
precision   0.82  0.61      0.74       0.71          0.74
recall      0.77  0.68      0.74       0.72          0.74
f1-score    0.79  0.64      0.74       0.72          0.74
support   151.00 80.00      0.74     231.00        231.00
'''

scores['Gradient Boosting'] = {
        'Train': accuracy_score(y_train, grad_boost_clf.predict(X_train)),
        'Test': accuracy_score(y_test, grad_boost_clf.predict(X_test)),
    }


################Voting#############
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

estimators = []
log_reg = LogisticRegression(solver='liblinear')
estimators.append(('Logistic', log_reg))

tree = DecisionTreeClassifier()
estimators.append(('Tree', tree))

svm_clf = SVC(gamma='scale')
estimators.append(('SVM', svm_clf))

voting = VotingClassifier(estimators=estimators)
voting.fit(X_train, y_train)

evaluate(voting, X_train, X_test, y_train, y_test)
'''
TRAINIG RESULTS: 
===============================
CONFUSION MATRIX:
[[327  22]
 [ 82 106]]
ACCURACY SCORE:
0.8063
CLASSIFICATION REPORT:
               0      1  accuracy  macro avg  weighted avg
precision   0.80   0.83      0.81       0.81          0.81
recall      0.94   0.56      0.81       0.75          0.81
f1-score    0.86   0.67      0.81       0.77          0.80
support   349.00 188.00      0.81     537.00        537.00
TESTING RESULTS: 
===============================
CONFUSION MATRIX:
[[131  20]
 [ 36  44]]
ACCURACY SCORE:
0.7576
CLASSIFICATION REPORT:
               0     1  accuracy  macro avg  weighted avg
precision   0.78  0.69      0.76       0.74          0.75
recall      0.87  0.55      0.76       0.71          0.76
f1-score    0.82  0.61      0.76       0.72          0.75
support   151.00 80.00      0.76     231.00        231.00
'''

scores['Voting'] = {
        'Train': accuracy_score(y_train, voting.predict(X_train)),
        'Test': accuracy_score(y_test, voting.predict(X_test)),
    }


scores_df = pd.DataFrame(scores)

scores_df.plot(kind='barh', figsize=(15, 8))
#################################################################################