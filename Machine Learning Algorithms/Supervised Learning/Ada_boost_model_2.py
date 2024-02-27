# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 21:02:24 2024

@author: adity
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,cross_val_score,kFold
from sklearn.metrics import accuracy_score,classification_report

data = pd.read_csv("C:/4-ML/Supervised Learning/Datasets/movies_classification.csv")

#data information
data.head()
data.info()
data.isna().sum()
#EDA
target=data["Start_Tech_Oscar"]
sns.countplot(x=target,palette='winter')
plt.xlabel("Oscar Rate");
#Our data is evenly distributed.Atleast 200 are there in both choice
plt.figure(figsize=(16,8))
sns.heatmap(data.corr(),annot=True,cmap='YLGnBu',fmt='.2f')
#Observations:

#1) Lead_Actor_Rating,Lead_Actress_rating,Director_rating and Producer_rating
sns.countplot(x='Genre',data=data,hue='Start_Tech_Oscar',palette='pastel')
plt.title('O chance based on Ticket Class',fontsize=10)
#Observations:
#Here are more chances of getting Oscar in Drama,comedy and action genre
sns.countplot(x='3D_available',daa=data,hue='Start_Tech_Oscar',palette='patel')
plt.title('O chance based on Ticket Class',fontsize=10)
#Observation:
#It is clear from plot that if #D is available the there is chance of
sns.set_context('notebook',font_scale=1.2)
fig,ax=plt.subplots(2,figsize=(20,13))

plt.suptitle('Distribution of Twitter_hastags and Collection based on target variable',fontsize=20)

ax1=sns.histplot(x='Twitter_hastags',data=data,hue='Start_Tech_Oscar',kde=True,ax=ax[0],palette='winter')
ax1.set(xlabel='Twitter_hastags',title='Distribution of Twitter_hastags based on target variable')

ax2=sns.histplot(x='Collection',data=data,hue="Start_Tech_Oscar",kde=True,ax=ax[1],palette='viridis')
ax2.set(xlabel='Collection',title='Distribution of Fare based on target variable')

plt.show()
data.hist(bins=30,figsize=(20,15),color='#005b96')

#As we can see there are outliers in Twitter_hastags,Marketing expense,Time taken
sns.boxplot(x=data['Twitter_hastags'])
sns.boxplot(x=data['Marketing expense'])
sns.boxplot(x=data['Time_taken'])
sns.boxplot(x=data['Avg_age_actors'])
#write code for winsorizor
#checking skewness
skew_df=pd.DataFrame(data.select_dtypes(np.number).columns,columns=['Feature'])
skew_df['Skew']=skew_df['Feature'].apply(lambda feature:skew(data[feature]))
skew_df['Absolute Skew']=skew_df['Skew'].apply(abs)
skew_df['skewed']==skew_df['Absolute Skew'].apply(lambda x:True if x>=0.5 else False)
skew_df
#Total charges column is clearly skewed as we also saw in the histogram,
for column in skew_df.query("Skewed==True")['Feature'].values:
    data[column]=np.log1p(data[column])

data.head()
#Encoding
data1=data.copy()
data1=pd.get_dummies(data1)

data1.head()
#scaling
data2=data1.copy()
sc=StandardScaler()
data2[data1.select_dtypes(np.number).columns]=sc.fit_transform(data2[data1.select_dtypes(np.number).columns])
data2.drop(['Start_Tech_Oscar'],axis=1,inplace=True)
data2.head()

#Splitting
data_f=data2.copy()
target=data['Start_Tech_Oscar']
target=target.astype(int)
target
X_train,X_test,y_train,y_test=train_test_split(data_f,target,test_size=0.2,stratify=target,random_state=42)
#Modelling
from sklearn.ensemble import AdaBoostClassifier

ada_clf=AdaBoostClassifier(learning_rate=0.02,n_estimators=5000)
ada_clf.fit(X_train,y_train)

from sklearn.metrics import accuracy_score,confusion_matrix

#Evaluation on testing data
confusion_matrix(y_test,ada_clf.predict(X_test))
accuracy_score(y_test,ada_clf.predict(X_test))

#Evaluation on training data
accuracy_score(y_train,ada_clf.predict(X_train))

from sklearn.metrics import accuracy_score,confusion_matrix