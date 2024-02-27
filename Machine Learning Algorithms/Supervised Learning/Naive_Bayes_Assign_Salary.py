# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 15:41:40 2024

Prepare a classification model using the Naive Bayes algorithm for 
the salary dataset. Train and test datasets are given separately. 
Use both for model building. 

@author: aditya
-------------------------------------------------------------------------------
business problem
    What is business objective?
    1.1. The motivation of human objective in a business is to find ways to 
         meet the needs of your employees, So that they feel valued and supported.
    1.2. Organic business objectives are goals that incorporate all aspects 
         its development, survival, progress and outlook.
    Are there any constraints?
    1.3. Information associated with each job post as text and then they represent 
         a consequence, this vector are often characterized by very high number of 
         dimensions(in the order of thousands)
         therefore it is necessary to collect huge amount of job post to be able to
-------------------------------------------------------------------------------
Data Description:

age -- age of a person
workclass -- A work class is a grouping of work
education -- Education of an individuals
maritalstatus -- Marital status of an individulas
occupation -- occupation of an individuals
relationship --
race -- Race of an Individual
sex -- Gender of an Individual
capitalgain -- profit received from the sale of an investment
capitalloss -- A decrease in the value of a capital asset
hoursperweek -- number of hours work per week
native -- Native of an individual
Salary -- salary of an individual
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.preprocessing import StandardScaler
#loading data
df_train = pd.read_csv("C:/4-ML/Supervised Learning/SalaryData_Train.csv",encoding='ISO-8859-1')
df_test = pd.read_csv("C:/4-ML/Supervised Learning/SalaryData_Test.csv",encoding='ISO-8859-1')

#EDA & Data Preprocessing
df_train.shape
#(30161, 14)

df_train.dtypes
'''
age               int64
workclass        object
education        object
educationno       int64
maritalstatus    object
occupation       object
relationship     object
race             object
sex              object
capitalgain       int64
capitalloss       int64
hoursperweek      int64
native           object
Salary           object
dtype: object
'''

a=pd.isnull(df_train)
a.sum()
#There is no value in the dataset

q=df_train.value_counts('Salary')
q
#Salary
'''
 <=50K    22653
 >50K      7508
dtype: int64
'''

v=df_train.describe()
v
# we get 5 number sumarry
'''
                age   educationno   capitalgain   capitalloss  hoursperweek
count  30161.000000  30161.000000  30161.000000  30161.000000  30161.000000
mean      38.438115     10.121316   1092.044064     88.302311     40.931269
std       13.134830      2.550037   7406.466611    404.121321     11.980182
min       17.000000      1.000000      0.000000      0.000000      1.000000
25%       28.000000      9.000000      0.000000      0.000000     40.000000
50%       37.000000     10.000000      0.000000      0.000000     40.000000
75%       47.000000     13.000000      0.000000      0.000000     45.000000
max       90.000000     16.000000  99999.000000   4356.000000     99.000000
'''

df_train.info()
# This will give us the information about all the points
'''
RangeIndex: 30161 entries, 0 to 30160
Data columns (total 14 columns):
 #   Column         Non-Null Count  Dtype 
---  ------         --------------  ----- 
 0   age            30161 non-null  int64 
 1   workclass      30161 non-null  object
 2   education      30161 non-null  object
 3   educationno    30161 non-null  int64 
 4   maritalstatus  30161 non-null  object
 5   occupation     30161 non-null  object
 6   relationship   30161 non-null  object
 7   race           30161 non-null  object
 8   sex            30161 non-null  object
 9   capitalgain    30161 non-null  int64 
 10  capitalloss    30161 non-null  int64 
 11  hoursperweek   30161 non-null  int64 
 12  native         30161 non-null  object
 13  Salary         30161 non-null  object
dtypes: int64(5), object(9)
memory usage: 3.2+ MB
'''

df_train_encoded = pd.get_dummies(df_train)
df_train_encoded
#This replaced the categorical value with binary numbers 0 or 1

# frequency for categorical fields 
category_col =['workclass', 'education','maritalstatus', 'occupation', 'relationship', 'race', 'sex', 'native', 'Salary'] 
for c in category_col:
    print (c)
    print (df_train[c].value_counts())
    print('\n')

'''
workclass
 Private             22285
 Self-emp-not-inc     2499
 Local-gov            2067
 State-gov            1279
 Self-emp-inc         1074
 Federal-gov           943
 Without-pay            14
Name: workclass, dtype: int64


education
 HS-grad         9840
 Some-college    6677
 Bachelors       5044
 Masters         1627
 Assoc-voc       1307
 11th            1048
 Assoc-acdm      1008
 10th             820
 7th-8th          557
 Prof-school      542
 9th              455
 12th             377
 Doctorate        375
 5th-6th          288
 1st-4th          151
 Preschool         45
Name: education, dtype: int64


maritalstatus
 Married-civ-spouse       14065
 Never-married             9725
 Divorced                  4214
 Separated                  939
 Widowed                    827
 Married-spouse-absent      370
 Married-AF-spouse           21
Name: maritalstatus, dtype: int64


occupation
 Prof-specialty       4038
 Craft-repair         4030
 Exec-managerial      3992
 Adm-clerical         3721
 Sales                3584
 Other-service        3212
 Machine-op-inspct    1965
 Transport-moving     1572
 Handlers-cleaners    1350
 Farming-fishing       989
 Tech-support          912
 Protective-serv       644
 Priv-house-serv       143
 Armed-Forces            9
Name: occupation, dtype: int64


relationship
 Husband           12463
 Not-in-family      7726
 Own-child          4466
 Unmarried          3212
 Wife               1406
 Other-relative      888
Name: relationship, dtype: int64


race
 White                 25932
 Black                  2817
 Asian-Pac-Islander      895
 Amer-Indian-Eskimo      286
 Other                   231
Name: race, dtype: int64


sex
 Male      20380
 Female     9781
Name: sex, dtype: int64


native
 United-States                 27504
 Mexico                          610
 Philippines                     188
 Germany                         128
 Puerto-Rico                     109
 Canada                          107
 India                           100
 El-Salvador                     100
 Cuba                             92
 England                          86
 Jamaica                          80
 South                            71
 China                            68
 Italy                            68
 Dominican-Republic               67
 Vietnam                          64
 Guatemala                        63
 Japan                            59
 Poland                           56
 Columbia                         56
 Iran                             42
 Taiwan                           42
 Haiti                            42
 Portugal                         34
 Nicaragua                        33
 Peru                             30
 Greece                           29
 France                           27
 Ecuador                          27
 Ireland                          24
 Hong                             19
 Cambodia                         18
 Trinadad&Tobago                  18
 Laos                             17
 Thailand                         17
 Yugoslavia                       16
 Outlying-US(Guam-USVI-etc)       14
 Hungary                          13
 Honduras                         12
 Scotland                         11
Name: native, dtype: int64


Salary
 <=50K    22653
 >50K      7508
Name: Salary, dtype: int64
'''
###########################################################################3
##########################Visualization############################
import seaborn as sns
sns.heatmap(df_train.isnull())


# countplot for all categorical columns
import matplotlib.pyplot as plt
sns.set(rc={'figure.figsize':(9,8)})
cat_col = ['workclass', 'education','maritalstatus', 'occupation', 'relationship', 'race', 'sex','Salary']
for col in cat_col:
    plt.figure() #this creates a new figure on which your plot will appear
    sns.countplot(x = col, data = df_train, palette = 'Set3');
    
##############################################################
#Printing unique values from each categorical columns
print('workclass',df_train.workclass.unique())
print('education',df_train.education.unique())
print('maritalstatus',df_train['maritalstatus'].unique())
print('occupation',df_train.occupation.unique())
print('relationship',df_train.relationship.unique())
print('race',df_train.race.unique())
print('sex',df_train.sex.unique())
print('native',df_train['native'].unique())
print('Salary',df_train.Salary.unique())
'''
workclass [' State-gov' ' Self-emp-not-inc' ' Private' ' Federal-gov' ' Local-gov'
 ' Self-emp-inc' ' Without-pay']
education [' Bachelors' ' HS-grad' ' 11th' ' Masters' ' 9th' ' Some-college'
 ' Assoc-acdm' ' 7th-8th' ' Doctorate' ' Assoc-voc' ' Prof-school'
 ' 5th-6th' ' 10th' ' Preschool' ' 12th' ' 1st-4th']
maritalstatus [' Never-married' ' Married-civ-spouse' ' Divorced'
 ' Married-spouse-absent' ' Separated' ' Married-AF-spouse' ' Widowed']
occupation [' Adm-clerical' ' Exec-managerial' ' Handlers-cleaners' ' Prof-specialty'
 ' Other-service' ' Sales' ' Transport-moving' ' Farming-fishing'
 ' Machine-op-inspct' ' Tech-support' ' Craft-repair' ' Protective-serv'
 ' Armed-Forces' ' Priv-house-serv']
relationship [' Not-in-family' ' Husband' ' Wife' ' Own-child' ' Unmarried'
 ' Other-relative']
race [' White' ' Black' ' Asian-Pac-Islander' ' Amer-Indian-Eskimo' ' Other']
sex [' Male' ' Female']
native [' United-States' ' Cuba' ' Jamaica' ' India' ' Mexico' ' Puerto-Rico'
 ' Honduras' ' England' ' Canada' ' Germany' ' Iran' ' Philippines'
 ' Poland' ' Columbia' ' Cambodia' ' Thailand' ' Ecuador' ' Laos'
 ' Taiwan' ' Haiti' ' Portugal' ' Dominican-Republic' ' El-Salvador'
 ' France' ' Guatemala' ' Italy' ' China' ' South' ' Japan' ' Yugoslavia'
 ' Peru' ' Outlying-US(Guam-USVI-etc)' ' Scotland' ' Trinadad&Tobago'
 ' Greece' ' Nicaragua' ' Vietnam' ' Hong' ' Ireland' ' Hungary']
Salary [' <=50K' ' >50K']
'''

df_train[['Salary', 'age']].groupby(['Salary'], as_index=False).mean().sort_values(by='age', ascending=False)
'''
   Salary        age
1    >50K  43.959110
0   <=50K  36.608264
'''

plt.style.use('seaborn-whitegrid')
x, y, hue = "race", "prop", "sex"
#hue_order = ["Male", "Female"]
plt.figure(figsize=(20,10)) 
f, axes = plt.subplots(1, 2)
sns.countplot(x=x, hue=hue, data=df_train, ax=axes[0])

prop_df = (df_train[x]
           .groupby(df_train[hue])
           .value_counts(normalize=True)
           .rename(y)
           .reset_index())

sns.barplot(x=x, y=y, hue=hue, data=prop_df, ax=axes[1])
##############################

g = sns.jointplot(x = 'age', 
              y = 'hoursperweek',
              data = df_train, 
              kind = 'hex', 
              cmap= 'hot', 
              size=10)

sns.regplot(df_train.age, df_train['hoursperweek'], ax=g.ax_joint, scatter=False, color='grey')
############################################################################################

#Feature encoding

from sklearn.preprocessing import LabelEncoder

df_train = df_train.apply(LabelEncoder().fit_transform)
df_train.head()

df_test = df_test.apply(LabelEncoder().fit_transform)
df_test.head()
#######################################################

#Test-Train-Split
from sklearn.model_selection import train_test_split

drop_elements = ['education', 'native', 'Salary']
X = df_train.drop(drop_elements, axis=1)

y = df_train['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
##########################################################################
#################Building Multinomial Naive Bays Model##########################

# Preparing a naive bayes model on training data set 

from sklearn.naive_bayes import MultinomialNB as MB

# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(X_train, y_train)

score_multinomial_train = classifier_mb.score(X_train,y_train)
score_multinomial_train
#0.7788390161825111


score_multinomial = classifier_mb.score(X_test,y_test)
score_multinomial
#0.7796865581675708

##Testing Multinomial Naive Bayes model on SalaryData_Test.csv
from sklearn import metrics

drop_elements = ['education', 'native', 'Salary']
X_new = df_test.drop(drop_elements, axis=1)

y_new = df_test['Salary']

# make predictions
new_prediction = classifier_mb.predict(X_new)
# summarize the fit of the model
print(metrics.classification_report(y_new, new_prediction))
print(metrics.confusion_matrix(y_new, new_prediction))
'''
              precision    recall  f1-score   support

           0       0.80      0.94      0.87     11360
           1       0.61      0.30      0.40      3700

    accuracy                           0.78     15060
   macro avg       0.71      0.62      0.63     15060
weighted avg       0.76      0.78      0.75     15060

[[10648   712]
 [ 2587  1113]]
'''

print("Accuracy:",metrics.accuracy_score(y_new, new_prediction))
print("Precision:",metrics.precision_score(y_new, new_prediction))
print("Recall:",metrics.recall_score(y_new, new_prediction))






