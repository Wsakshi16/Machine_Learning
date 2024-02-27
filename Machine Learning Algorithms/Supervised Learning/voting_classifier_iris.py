# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 09:02:18 2024

@author: aditya
"""

import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection

import warnings
warnings.filterwarnings('ignore')
from sklearn import datasets
iris = datasets.load_iris()
X,y = iris.data[:,1:3],iris.target #taking entire dataset
clf1 = LogisticRegression()
clf2 = RandomForestClassifier()
clf3 = GaussianNB()

print("After three fold cross validation")
labels = ['Logistic Regression','Random Forest model','Naive bayes model']
for clf, label in zip([clf1,clf2,clf3],labels):
    scores=model_selection.cross_val_score(clf,X,y,cv=5,scoring='accuracy')
    print('Accuracy: ',scores.mean(),'for ',label)
'''
After three fold cross validation
Accuracy:  0.9533333333333334 for  Logistic Regression
Accuracy:  0.9466666666666667 for  Random Forest model
Accuracy:  0.9133333333333334 for  Naive bayes model
'''

voting_clf_hard = VotingClassifier(estimators=[(label[0],clf1),
                                               (label[1],clf2),
                                               (label[2],clf3)],
                                               voting='hard')
    
voting_clf_soft = VotingClassifier(estimators=[(label[0],clf1),
                                               (label[1],clf2),
                                               (label[2],clf3)],
                                               voting='soft')

labels_new = ['Logistic Regression','Random Forest model','Naive Bayes model','Hard Voting','Soft Voting']
for clf, label in zip([clf1,clf2,clf3,voting_clf_hard,voting_clf_soft],labels_new):
    scores=model_selection.cross_val_score(clf,X,y,cv=5,scoring='accuracy')
    print('Accuracy: ',scores.mean(),"for ",label)
'''
Accuracy:  0.9533333333333334 for  Logistic Regression
Accuracy:  0.9466666666666667 for  Random Forest model
Accuracy:  0.9133333333333334 for  Naive Bayes model
Accuracy:  0.9466666666666667 for  Hard Voting
Accuracy:  0.9466666666666667 for  Soft Voting
'''