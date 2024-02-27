# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 13:44:15 2024

@author: adity
"""

'''
Problem Statement:
Data privacy is always an important factor to safeguard their customers' details. 
For this, password strength is an important metric to track. Build an ensemble 
model to classify the user’s password strength.
'''

import pandas as pd
import matplotlib.pyplot as plt
# For text feature extraction
from sklearn.feature_extraction.text import TfidfVectorizer
# For creating a pipeline
from sklearn.pipeline import Pipeline
# Classifier Model (Decision Tree)
from sklearn.tree import DecisionTreeClassifier
# Read the File
data = pd.read_excel('C:/4-ML/Supervised Learning/Datasets/Ensemble_Password_Strength.xlsx')
data.describe()
'''
       characters_strength
count          1999.000000
mean              0.857929
std               0.349210
min               0.000000
25%               1.000000
50%               1.000000
75%               1.000000
max               1.000000
'''

data.head()
'''
    characters  characters_strength
0     kzde5577                    1
1     kino3434                    1
2    visi7k1yr                    1
3     megzy123                    1
4  lamborghin1                    1
'''
# Features which are passwords
# Selecting all rows and coloumn 1 which are passwords of type 'string'.

features = data.values[:, 0].astype('str')
print(features)
'''
['kzde5577' 'kino3434' 'visi7k1yr' ... 'marco90' 'jebekk1' 'akosi091692']
'''
# Labels which are strength of password
# Selecting all rows and last coloumn which are passwords strengths of type 'int'.

labels = data.values[:, 1].astype('int')
print(labels)
#[1 1 1 ... 0 0 1]

# Sequentially apply a list of transforms and a final estimator
classifier_model = Pipeline([
                ('tfidf', TfidfVectorizer(analyzer='char')),
                ('decisionTree',DecisionTreeClassifier()),
])

# Fit the Model
classifier_model.fit(features, labels)
# Instead of splitting dataset into training and testing, we keep test dataset as seprate .csv file 
#df= pd.read_csv('C:\\Users\\ankush\\Desktop\\cleanpasswordlist.csv')

X = df.values[:,0].astype('str')
y = df.values[:, 1].astype('int')
print('Testing Accuracy: ',classifier_model.score(X, y)*100)
#showing predication for 50 passwords as a sample

list=features[40:90]
predict=classifier_model.predict(list)
predict
print(list)
# Taking sample of 50 passwords for ploting on Graph

x=features[100:150]
y=classifier_model.predict(x)

# Ploting graph

plt.scatter(x, y, color = 'red')
plt.title('Password vs Strength')
plt.xlabel('Password String')
plt.ylabel('Strength scale')
plt.show()
# Printing x coordinate

print(x)
# Printing y coordinate

print(y)