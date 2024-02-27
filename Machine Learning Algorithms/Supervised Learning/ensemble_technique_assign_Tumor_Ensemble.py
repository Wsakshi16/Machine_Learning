# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 13:03:30 2024

@author: adity
"""

"""
Problem Statement:

"""
# compare standalone models for binary classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from matplotlib import pyplot
import pandas as pd
import numpy as np

wbcd = pd.read_csv("C:/4-ML/Supervised Learning/Datasets/Tumor_Ensemble.csv")

# converting B to Benign and M to Malignant 
wbcd['diagnosis'] = np.where(wbcd['diagnosis'] == 'B', 'Benign ', wbcd['diagnosis'])
wbcd['diagnosis'] = np.where(wbcd['diagnosis'] == 'M', 'Malignant ', wbcd['diagnosis'])

wbcd = wbcd.iloc[:, 1:32] # Excluding id column

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
wbcd_n = norm_func(wbcd.iloc[:, 1:])
wbcd_n.describe()
'''
       radius_mean  texture_mean  ...  symmetry_worst  dimension_worst
count   569.000000    569.000000  ...      569.000000       569.000000
mean      0.338222      0.323965  ...        0.263307         0.189596
std       0.166787      0.145453  ...        0.121954         0.118466
min       0.000000      0.000000  ...        0.000000         0.000000
25%       0.223342      0.218465  ...        0.185098         0.107700
50%       0.302381      0.308759  ...        0.247782         0.163977
75%       0.416442      0.408860  ...        0.318155         0.242949
max       1.000000      1.000000  ...        1.000000         1.000000

[8 rows x 30 columns]
'''

X = np.array(wbcd_n.iloc[:,:]) # Predictors 
y = np.array(wbcd['diagnosis']) # Target 

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2)



###################Stacking##################

# get a list of models to evaluate
def get_models():
	models = dict()
	models['lr'] = LogisticRegression()
	models['knn'] = KNeighborsClassifier()
	models['cart'] = DecisionTreeClassifier()
	models['svm'] = SVC()
	models['bayes'] = GaussianNB()
	return models

# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
	return scores


# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	scores = evaluate_model(model, X, y)
	results.append(scores)
	names.append(name)
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
'''
std(scores)))
>lr 0.968 (0.023)
>knn 0.966 (0.023)
>cart 0.932 (0.023)
>svm 0.977 (0.017)
>bayes 0.934 (0.032)
'''

# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()


# compare ensemble to each baseline classifier
from sklearn.ensemble import StackingClassifier
from matplotlib import pyplot

# get a stacking ensemble of models
def get_stacking():
	# define the base models
	level0 = list()
	level0.append(('lr', LogisticRegression()))
	level0.append(('knn', KNeighborsClassifier()))
	level0.append(('cart', DecisionTreeClassifier()))
	level0.append(('svm', SVC()))
	level0.append(('bayes', GaussianNB()))
	# define meta learner model
	level1 = LogisticRegression()
	# define the stacking ensemble
	model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
	return model

# get a list of models to evaluate
def get_models():
	models = dict()
	models['lr'] = LogisticRegression()
	models['knn'] = KNeighborsClassifier()
	models['cart'] = DecisionTreeClassifier()
	models['svm'] = SVC()
	models['bayes'] = GaussianNB()
	models['stacking'] = get_stacking()
	return models

# evaluate a give model using cross-validation
def evaluate_model(model, X, y):
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
	return scores


# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	scores = evaluate_model(model, X, y)
	results.append(scores)
	names.append(name)
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
'''
std(scores)))
>lr 0.968 (0.023)
>knn 0.966 (0.023)
>cart 0.931 (0.030)
>svm 0.977 (0.017)
>bayes 0.934 (0.032)
>stacking 0.972 (0.019)
'''

# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
##################### Evaluate function #####################

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
    
######################BAGGING#####################
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    
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
[[292   0]
 [  0 163]]
ACCURACY SCORE:
1.0000
CLASSIFICATION REPORT:
           Benign   Malignant   accuracy  macro avg  weighted avg
precision      1.0         1.0       1.0        1.0           1.0
recall         1.0         1.0       1.0        1.0           1.0
f1-score       1.0         1.0       1.0        1.0           1.0
support      292.0       163.0       1.0      455.0         455.0
TESTING RESULTS: 
===============================
CONFUSION MATRIX:
[[64  1]
 [ 2 47]]
ACCURACY SCORE:
0.9737
CLASSIFICATION REPORT:
             Benign   Malignant   accuracy   macro avg  weighted avg
precision   0.969697    0.979167  0.973684    0.974432      0.973767
recall      0.984615    0.959184  0.973684    0.971900      0.973684
f1-score    0.977099    0.969072  0.973684    0.973086      0.973649
support    65.000000   49.000000  0.973684  114.000000    114.000000
'''

scores = {
    'Bagging Classifier': {
        'Train': accuracy_score(y_train, bagging_clf.predict(X_train)),
        'Test': accuracy_score(y_test, bagging_clf.predict(X_test)),
    },
}
################Ada Boosting#############
from sklearn.ensemble import AdaBoostClassifier

ada_boost_clf = AdaBoostClassifier(n_estimators=30)
ada_boost_clf.fit(X_train, y_train)
evaluate(ada_boost_clf, X_train, X_test, y_train, y_test)
'''
TRAINIG RESULTS: 
===============================
CONFUSION MATRIX:
[[292   0]
 [  0 163]]
ACCURACY SCORE:
1.0000
CLASSIFICATION REPORT:
           Benign   Malignant   accuracy  macro avg  weighted avg
precision      1.0         1.0       1.0        1.0           1.0
recall         1.0         1.0       1.0        1.0           1.0
f1-score       1.0         1.0       1.0        1.0           1.0
support      292.0       163.0       1.0      455.0         455.0
TESTING RESULTS: 
===============================
CONFUSION MATRIX:
[[64  1]
 [ 2 47]]
ACCURACY SCORE:
0.9737
CLASSIFICATION REPORT:
             Benign   Malignant   accuracy   macro avg  weighted avg
precision   0.969697    0.979167  0.973684    0.974432      0.973767
recall      0.984615    0.959184  0.973684    0.971900      0.973684
f1-score    0.977099    0.969072  0.973684    0.973086      0.973649
support    65.000000   49.000000  0.973684  114.000000    114.000000
'''

scores['AdaBoost'] = {
        'Train': accuracy_score(y_train, ada_boost_clf.predict(X_train)),
        'Test': accuracy_score(y_test, ada_boost_clf.predict(X_test)),
    }

################GradientBoosting#############

from sklearn.ensemble import GradientBoostingClassifier

grad_boost_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
grad_boost_clf.fit(X_train, y_train)
evaluate(grad_boost_clf, X_train, X_test, y_train, y_test)
'''
TRAINIG RESULTS: 
===============================
CONFUSION MATRIX:
[[292   0]
 [  0 163]]
ACCURACY SCORE:
1.0000
CLASSIFICATION REPORT:
           Benign   Malignant   accuracy  macro avg  weighted avg
precision      1.0         1.0       1.0        1.0           1.0
recall         1.0         1.0       1.0        1.0           1.0
f1-score       1.0         1.0       1.0        1.0           1.0
support      292.0       163.0       1.0      455.0         455.0
TESTING RESULTS: 
===============================
CONFUSION MATRIX:
[[64  1]
 [ 2 47]]
ACCURACY SCORE:
0.9737
CLASSIFICATION REPORT:
             Benign   Malignant   accuracy   macro avg  weighted avg
precision   0.969697    0.979167  0.973684    0.974432      0.973767
recall      0.984615    0.959184  0.973684    0.971900      0.973684
f1-score    0.977099    0.969072  0.973684    0.973086      0.973649
support    65.000000   49.000000  0.973684  114.000000    114.000000
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
[[292   0]
 [  7 156]]
ACCURACY SCORE:
0.9846
CLASSIFICATION REPORT:
              Benign   Malignant   accuracy   macro avg  weighted avg
precision    0.976589    1.000000  0.984615    0.988294      0.984976
recall       1.000000    0.957055  0.984615    0.978528      0.984615
f1-score     0.988156    0.978056  0.984615    0.983106      0.984538
support    292.000000  163.000000  0.984615  455.000000    455.000000
TESTING RESULTS: 
===============================
CONFUSION MATRIX:
[[65  0]
 [ 2 47]]
ACCURACY SCORE:
0.9825
CLASSIFICATION REPORT:
             Benign   Malignant   accuracy   macro avg  weighted avg
precision   0.970149    1.000000  0.982456    0.985075      0.982980
recall      1.000000    0.959184  0.982456    0.979592      0.982456
f1-score    0.984848    0.979167  0.982456    0.982008      0.982406
support    65.000000   49.000000  0.982456  114.000000    114.000000
'''

scores['Voting'] = {
        'Train': accuracy_score(y_train, voting.predict(X_train)),
        'Test': accuracy_score(y_test, voting.predict(X_test)),
    }


scores_df = pd.DataFrame(scores)

scores_df.plot(kind='barh', figsize=(15, 8))
#######################################################################