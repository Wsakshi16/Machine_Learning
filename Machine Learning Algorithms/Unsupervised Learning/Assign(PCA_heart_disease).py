# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 16:35:39 2023

@author: adity
"""
#Heart_disease

'''
STEP 1:
Dataset Explanation: It contains 76 attributes, including the predicted attribute, 
but all published experiments refer to using a subset of 14 of them. 
The "target" field refers to the presence of heart disuhiease in the patient. 
It is integer valued 0 => no disease and 1 => disease.
-------------------------------------------------------------------------------
Problem Statement: A pharmaceuticals manufacturing company is conducting a study 
on a new medicine to treat heart diseases. The company has gathered data from 
its secondary sources and would like you to provide high level analytical 
insights on the data. Its aim is to segregate patients depending on their age 
group and other factors given in the data. Perform PCA and clustering algorithms 
on the dataset and check if the clusters formed before and after PCA are the 
same and provide a brief report on your model. You can also explore more ways 
to improve your model. 
-------------------------------------------------------------------------------
business objective: The primary objective of the pharmaceuticals manufacturing 
company is to gain actionable insights from the collected data to enhance the 
development and targeted marketing of the new medicine for heart diseases.

Maximize: Maximize the precision in targeting specific patient groups. 
The company aims to identify subgroups of patients with similar characteristics 
to optimize marketing strategies and treatment approaches.
    
Minimize: 1)Minimize the loss of information during dimensionality reduction through PCA.
          2)Minimize misclassification or ambiguity in the clusters formed by the 
          clustering algorithms.

constraints: Data Privacy and Compliance, Continuous Improvement.
----------------------------------------------------------------------------------------
STEP 2:
#data dictionary

Name of feature       Description                     Type          Relevance
    age        - age in years                    -   Integer     -  Relevant
    sex        - sex (1 = male;                  -   Categorical -  Relevant
                     0 = female) 
    cp         - chest pain type                 -   Categorical -  Relevant
  trestbps     - resting blood pressure          -   Integer     -  Relevant
                 (on admission to the hospital)
   chol        - serum cholestoral               -   Integer     -  Relevant
   fbs         - fasting blood sugar > 120 mg/dl -   Categorical -  Relevant
  restecg      - resting electrocardiographic    -   Categorical -  Relevant
                 results
  thalach      - maximum heart rate achieved     -   Integer     -  Relevant
   exang       - exercise induced angina         -   Categorical -  Relevant
  oldpeak      - ST depression induced by        -   Integer     -  Relevant
                 exercise relative to rest
   slope       - the slope of the peak exercise  -   Categorical -  Relevant
                 ST segment  
    ca         - number of major vessels (0-3)   -   Integer     -  Relevant
                 colored by flourosopy 
   thal        - 3 = normal; 6 = fixed defect    -   Categorical -  Relevant
                 7 = reversable defect
  target       - diagnosis of heart disease      -   Integer     - Relevant


'''

import numpy as np
import pandas as pd
heart_data = pd.read_csv("c:/0-datasets/heart disease.csv")
heart_data.describe()

#STEP 3:	Data Pre-processing
#Check the shape of the dataset
heart_data.shape
#(303, 14) that is there are 303 rows and 14 columns present in dataset

#Check the null value present in dataset
heart_data.isnull().sum()
#There is no any null value.

heart_data.info()

heart_data['restecg'].unique()
###################################################################
#STEP 4: EDA
# Check for the outlier
# Plot a boxplot to check whether it contain an outlier or not

import seaborn as sns
sns.boxplot(heart_data)
# the trestbps,chol and thalach column contain the outlier
####################
sns.histplot(heart_data)
# In this second and third columns showing the darkcolour so they are co-related to each other

###################################################################################################################################################
# So there is outlier is present and also the the column shows skewness
# and there is scale difference in mean and SD so there is need to scale the dataset
#Scaling the dataset
from sklearn.preprocessing import scale

X = heart_data.iloc[:,0:13].values
Y = heart_data.iloc[:,13].values

X.shape, Y.shape

from sklearn.preprocessing import scale
scaled_X = scale(X)
###########################################################################################
#STEP 5: Build a model
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
pca = PCA(n_components=13)


pca.fit(X)

def showVarianceRatio(pca):
    exp_ratio_var = pca.explained_variance_ratio_
    var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
    plt.plot(var1)

exp_ratio_var = pca.explained_variance_ratio_
exp_ratio_var


var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

plt.plot(var1)
#As per the plot above almost 100 % of variance is explained by the first 3 components

pca2 = PCA(n_components=3)
pca2.fit(X)
X1=pca2.fit_transform(X)


X1.shape

showVarianceRatio(pca2)

with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))
    for lab, col in zip((0, 1, 2, 3),
                        ('blue', 'red', 'green', 'black')):
        plt.scatter(X1[Y==lab, 0],
                    X1[Y==lab, 1],
                    label=lab,
                    c=col)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(loc='lower center')
    plt.tight_layout()
    plt.show()
    
    
