# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 13:19:34 2024

@author: adity
"""
'''
Problem Statement: A glass manufacturing plant uses different earth elements to 
design new glass materials based on customer requirements. For that, they would 
like to automate the process of classification as it’s a tedious job to manually 
classify them. Help the company achieve its objective by correctly classifying 
the glass type based on the other features using KNN algorithm.
-------------------------------------------------------------------------------------------
Dataset Description:

Id number: 1 to 214 (removed from CSV file)
RI: refractive index
Na: Sodium (unit measurement: weight percent in corresponding oxide, as are attributes 4-10)
Mg: Magnesium
Al: Aluminum
Si: Silicon
K: Potassium
Ca: Calcium
Ba: Barium
Fe: Iron
Type of glass: (class attribute)
-- 1 building_windows_float_processed
-- 2 building_windows_non_float_processed
-- 3 vehicle_windows_float_processed
-- 4 vehicle_windows_non_float_processed (none in this database)
-- 5 containers
-- 6 tableware
-- 7 headlamps
'''
import pandas as pd
import numpy as np

glass = pd.read_csv("C:/4-ML/Supervised Learning/Datasets/glass.csv")

# converting Type values to Glass Types
glass['Type'] = np.where(glass['Type'] == '0', 'Glass', glass['Type'])
glass['Type'] = np.where(glass['Type'] == '1', 'Glass 1', glass['Type'])
glass['Type'] = np.where(glass['Type'] == '2', 'Glass 2', glass['Type'])
glass['Type'] = np.where(glass['Type'] == '3', 'Glass 3', glass['Type'])
glass['Type'] = np.where(glass['Type'] == '4', 'Glass 4', glass['Type'])
glass['Type'] = np.where(glass['Type'] == '5', 'Glass 5', glass['Type'])
glass['Type'] = np.where(glass['Type'] == '6', 'Glass 6', glass['Type'])
glass['Type'] = np.where(glass['Type'] == '7', 'Glass 7', glass['Type'])

glass1 = glass.iloc[:, 0:9] # Excluding Type column

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
glass_n = norm_func(glass1.iloc[:, :])
glass_n.describe()

X = np.array(glass_n.iloc[:,:]) # Predictors 
Y = np.array(glass['Type']) # Target 

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3) 
knn.fit(X_train, Y_train)

pred = knn.predict(X_test)
pred

# Evaluate the model
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, pred))
pd.crosstab(Y_test, pred, rownames = ['Actual'], colnames= ['Predictions']) 


# error on train data
pred_train = knn.predict(X_train)
print(accuracy_score(Y_train, pred_train))
pd.crosstab(Y_train, pred_train, rownames=['Actual'], colnames = ['Predictions']) 


# creating empty list variable 
acc = []

# running KNN algorithm for 3 to 50 nearest neighbours(odd numbers) and 
# storing the accuracy values

for i in range(1,31,2):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train, Y_train)
    train_acc = np.mean(neigh.predict(X_train) == Y_train)
    test_acc = np.mean(neigh.predict(X_test) == Y_test)
    acc.append([train_acc, test_acc])


import matplotlib.pyplot as plt # library to do visualizations 

# train accuracy plot 
plt.plot(np.arange(1,31,2),[i[0] for i in acc],"ro-")

# test accuracy plot
plt.plot(np.arange(1,31,2),[i[1] for i in acc],"bo-")