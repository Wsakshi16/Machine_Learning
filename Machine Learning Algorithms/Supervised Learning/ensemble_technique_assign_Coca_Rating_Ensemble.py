# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 13:24:14 2024

@author: aditya
"""

"""
Problem Statement:
A sample of global companies and their ratings are given for the cocoa bean 
production along with the location of the beans being used. Identify the important 
features in the analysis and accurately classify the companies based on their 
ratings and draw insights from the data. Build ensemble models such as Bagging, 
Boosting, Stacking, and Voting on the dataset given.

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

cocoa = pd.read_excel("C:/4-ML/Supervised Learning/Datasets/Coca_Rating_Ensemble.xlsx")
cocoa.columns
'''
Index(['Company', 'Name', 'REF', 'Review', 'Cocoa_Percent', 'Company_Location',
       'Rating', 'Bean_Type', 'Origin'],
      dtype='object')
'''

cocoa.dtypes
'''
Index(['Company', 'Name', 'REF', 'Review', 'Cocoa_Percent', 'Company_Location',
       'Rating', 'Bean_Type', 'Origin'],
      dtype='object')
'''

cocoa.drop(['REF','Review','Name','Company'],axis=1, inplace=True)
cocoa1=cocoa.head()
'''
   Cocoa_Percent Company_Location  Rating Bean_Type    Origin
0           0.63           France    3.75            Sao Tome
1           0.70           France    2.75                Togo
2           0.70           France    3.00                Togo
3           0.70           France    3.50                Togo
4           0.70           France    3.50  
'''

cocoa1.plot(x="Company_Location", y=["Rating","Cocoa_Percent"], kind="bar", figsize=(9, 8))
 
cocoa1.show()

# Label Encoder
from sklearn.preprocessing import LabelEncoder
# creating instance of labelencoder
labelencoder = LabelEncoder()
cocoa['Company_Location']= labelencoder.fit_transform(cocoa['Company_Location'])
cocoa['Bean_Type']= labelencoder.fit_transform(cocoa['Bean_Type'])
cocoa['Origin']= labelencoder.fit_transform(cocoa['Origin'])

cocoa['Ratings'] = pd.cut(cocoa['Rating'], bins=[min(cocoa.Rating) - 1, 
                                                  cocoa.Rating.mean(), max(cocoa.Rating)], labels=["Low","High"])


cocoa.drop(['Rating'],axis=1, inplace=True)
from sklearn.preprocessing import MinMaxScaler


# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)



# Input and Output Split
predictors = cocoa.loc[:, cocoa.columns!="Ratings"]
type(predictors)
predictors1=norm_func(predictors)

target = cocoa["Ratings"]
type(target)

# Train Test partition of the data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(predictors1, target, test_size = 0.2, random_state=0)



###################Stacking################

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
def evaluate_model(model, predictors1, target):
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(model, predictors1, target, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
	return scores


# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
       scores = evaluate_model(model, predictors1, target)
       results.append(scores)
       names.append(name)
       print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
'''
>lr 0.573 (0.024)
>knn 0.566 (0.034)
>cart 0.575 (0.034)
>svm 0.578 (0.024)
>bayes 0.600 (0.026)
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
def evaluate_model(model, predictors1, target):
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(model, predictors1, target, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
	return scores


# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	scores = evaluate_model(model, predictors1, target)
	results.append(scores)
	names.append(name)
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
'''
>lr 0.573 (0.024)
>knn 0.566 (0.034)
>cart 0.578 (0.034)
>svm 0.578 (0.024)
>bayes 0.600 (0.026)
>stacking 0.598 (0.028)
'''

# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()

################################ Evaluate function ##############################

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(predictors1, target, test_size = 0.2)

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
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(predictors1, target, test_size = 0.2)
    
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
[[732  70]
 [ 91 543]]
ACCURACY SCORE:
0.8879
CLASSIFICATION REPORT:
                 High         Low  accuracy    macro avg  weighted avg
precision    0.889429    0.885808  0.887883     0.887618      0.887830
recall       0.912718    0.856467  0.887883     0.884593      0.887883
f1-score     0.900923    0.870890  0.887883     0.885907      0.887663
support    802.000000  634.000000  0.887883  1436.000000   1436.000000
TESTING RESULTS: 
===============================
CONFUSION MATRIX:
[[131  72]
 [ 77  79]]
ACCURACY SCORE:
0.5850
CLASSIFICATION REPORT:
                 High         Low  accuracy   macro avg  weighted avg
precision    0.629808    0.523179  0.584958    0.576493      0.583473
recall       0.645320    0.506410  0.584958    0.575865      0.584958
f1-score     0.637470    0.514658  0.584958    0.576064      0.584103
support    203.000000  156.000000  0.584958  359.000000    359.000000
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
evaluate(ada_boost_clf, X_train, X_test, y_train, y_test)
'''
TRAINIG RESULTS: 
===============================
CONFUSION MATRIX:
[[652 150]
 [374 260]]
ACCURACY SCORE:
0.6351
CLASSIFICATION REPORT:
                 High         Low  accuracy    macro avg  weighted avg
precision    0.635478    0.634146  0.635097     0.634812      0.634890
recall       0.812968    0.410095  0.635097     0.611531      0.635097
f1-score     0.713348    0.498084  0.635097     0.605716      0.618308
support    802.000000  634.000000  0.635097  1436.000000   1436.000000
TESTING RESULTS: 
===============================
CONFUSION MATRIX:
[[145  58]
 [107  49]]
ACCURACY SCORE:
0.5404
CLASSIFICATION REPORT:
                 High         Low  accuracy   macro avg  weighted avg
precision    0.575397    0.457944   0.54039    0.516670      0.524359
recall       0.714286    0.314103   0.54039    0.514194      0.540390
f1-score     0.637363    0.372624   0.54039    0.504993      0.522323
support    203.000000  156.000000   0.54039  359.000000    359.000000
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
[[686 116]
 [304 330]]
ACCURACY SCORE:
0.7075
CLASSIFICATION REPORT:
                 High         Low  accuracy    macro avg  weighted avg
precision    0.692929    0.739910  0.707521     0.716420      0.713672
recall       0.855362    0.520505  0.707521     0.687933      0.707521
f1-score     0.765625    0.611111  0.707521     0.688368      0.697406
support    802.000000  634.000000  0.707521  1436.000000   1436.000000
TESTING RESULTS: 
===============================
CONFUSION MATRIX:
[[146  57]
 [102  54]]
ACCURACY SCORE:
0.5571
CLASSIFICATION REPORT:
                 High         Low  accuracy   macro avg  weighted avg
precision    0.588710    0.486486  0.557103    0.537598      0.544290
recall       0.719212    0.346154  0.557103    0.532683      0.557103
f1-score     0.647450    0.404494  0.557103    0.525972      0.541876
support    203.000000  156.000000  0.557103  359.000000    359.000000
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
[[734  68]
 [409 225]]
ACCURACY SCORE:
0.6678
CLASSIFICATION REPORT:
                 High         Low  accuracy    macro avg  weighted avg
precision    0.642170    0.767918  0.667827     0.705044      0.697688
recall       0.915212    0.354890  0.667827     0.635051      0.667827
f1-score     0.754756    0.485437  0.667827     0.620096      0.635850
support    802.000000  634.000000  0.667827  1436.000000   1436.000000
TESTING RESULTS: 
===============================
CONFUSION MATRIX:
[[174  29]
 [117  39]]
ACCURACY SCORE:
0.5933
CLASSIFICATION REPORT:
                 High         Low  accuracy   macro avg  weighted avg
precision    0.597938    0.573529  0.593315    0.585734      0.587332
recall       0.857143    0.250000  0.593315    0.553571      0.593315
f1-score     0.704453    0.348214  0.593315    0.526334      0.549653
support    203.000000  156.000000  0.593315  359.000000    359.000000
'''

scores['Voting'] = {
        'Train': accuracy_score(y_train, voting.predict(X_train)),
        'Test': accuracy_score(y_test, voting.predict(X_test)),
    }


scores_df = pd.DataFrame(scores)

scores_df.plot(kind='barh', figsize=(15, 8))
############################################################################