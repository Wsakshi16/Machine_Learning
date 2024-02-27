# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 15:33:58 2023

@author: adity
"""
'''
Dataset Explanation: These data are the results of a chemical analysis of wines 
grown in the same region in Italy but derived from three different cultivars. 
The analysis determined the quantities of 13 constituents found in each of the 
three types of wines. 

Business Problem: Find the insight from the chemical analysis of wines.

business objective: 
maximize: A composite measure of wine quality, potentially derived from expert 
          ratings or sensory evaluations.

minimize: production cost.

constraints: Consistency of Brand Image
'''

import numpy as np
import pandas as pd
wine_dataset = pd.read_csv("c:/0-datasets/wine.csv")

#summary Statistics 
wine_dataset.describe() #It is also called as five number summary
'''
             Type     Alcohol       Malic  ...         Hue    Dilution      Proline
count  178.000000  178.000000  178.000000  ...  178.000000  178.000000   178.000000
mean     1.938202   13.000618    2.336348  ...    0.957449    2.611685   746.893258
std      0.775035    0.811827    1.117146  ...    0.228572    0.709990   314.907474
min      1.000000   11.030000    0.740000  ...    0.480000    1.270000   278.000000
25%      1.000000   12.362500    1.602500  ...    0.782500    1.937500   500.500000
50%      2.000000   13.050000    1.865000  ...    0.965000    2.780000   673.500000
75%      3.000000   13.677500    3.082500  ...    1.120000    3.170000   985.000000
max      3.000000   14.830000    5.800000  ...    1.710000    4.000000  1680.000000
'''
#Check the shape of the dataset
wine_dataset.shape
#(178, 14) that is there are 178 rows and 14 columns present in dataset

#Check the null value present in dataset
wine_dataset.isnull().sum()
#There is no any null value.
'''
Type               0
Alcohol            0
Malic              0
Ash                0
Alcalinity         0
Magnesium          0
Phenols            0
Flavanoids         0
Nonflavanoids      0
Proanthocyanins    0
Color              0
Hue                0
Dilution           0
Proline            0
dtype: int64
'''
wine_dataset.info()
#From this we will get to know the information about all the columns like null values, datatypes
'''
 #   Column           Non-Null Count  Dtype  
---  ------           --------------  -----  
 0   Type             178 non-null    int64  
 1   Alcohol          178 non-null    float64
 2   Malic            178 non-null    float64
 3   Ash              178 non-null    float64
 4   Alcalinity       178 non-null    float64
 5   Magnesium        178 non-null    int64  
 6   Phenols          178 non-null    float64
 7   Flavanoids       178 non-null    float64
 8   Nonflavanoids    178 non-null    float64
 9   Proanthocyanins  178 non-null    float64
 10  Color            178 non-null    float64
 11  Hue              178 non-null    float64
 12  Dilution         178 non-null    float64
 13  Proline          178 non-null    int64  
dtypes: float64(11), int64(3)
'''

# =============================================================================
# Data dictionary
# =============================================================================
'''
name of the feature    Description                   Type            Relevance
Type                - Type of wine                - Categorical - Irrelevant,Type does not provide useful information
Alcohol             - Quantity of Alcohol         - Continuous  - Relevant
                      present in wine             
Malic               - Quantity of Malic           - Continuous  - Relevant
Ash                 - Quantity of Ash             - Continuous  - Relevant
Alcalinity          - Quantity of Alcalinity      - Integer     - Relevant
Magnesium           - Quantity of Magnesium       - Continuous  - Relevant 
Phenols             - Quantity of Phenols         - Continuous  - Relevant
Flavanoids          - Quantity of Flavanoids      - Continuous  - Relevant  
Nonflavanoids       - Quantity of Nonflavanoids   - Continuous  - Relevant
Proanthocyanins     - Quantity of Proanthocyanins - Continuous  - Relevant
Color               - Quantity of color           - Continuous  - Relevant
Hue                 - Quantity of Hue             - Continuous  - Relevant
Dilution            - Quantity of Dilution        - Continuous  - Relevant
Proline             - Quantity of Proline         - Integer     - Relevant

'''

# =============================================================================
# Data Pre-processing
# =============================================================================
#Data Cleaning and Feature Engineering:
    
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from scipy import stats

#Boxplot: To check the outliers
plt.boxplot(wine_dataset['Alcohol'])
plt.boxplot(wine_dataset['Malic'])
plt.boxplot(wine_dataset['Ash'])
plt.boxplot(wine_dataset['Alcalinity'])
plt.boxplot(wine_dataset['Magnesium'])
plt.boxplot(wine_dataset['Phenols'])
plt.boxplot(wine_dataset['Flavanoids'])
plt.boxplot(wine_dataset['Nonflavanoids'])
plt.boxplot(wine_dataset['Proanthocyanins'])
plt.boxplot(wine_dataset['Color'])
plt.boxplot(wine_dataset['Hue'])
plt.boxplot(wine_dataset['Dilution'])
plt.boxplot(wine_dataset['Proline'])
#From above boxplot we can observe that there are 
#outliers present in some features such as ash,Alcalinity,Phenols,
#Proanthocyanins,Proanthocyanins,Hue,etc.

# Function to detect and handle outliers using Z-score
def handle_outliers_zscore(wine_dataset, threshold=3):
    z_scores = stats.zscore(wine_dataset)
    abs_z_scores = abs(z_scores)
    outliers = (abs_z_scores > threshold).all(axis=1)
    wine_dataset_no_outliers = wine_dataset[~outliers]
    return wine_dataset_no_outliers

# Handle outliers using Z-score
wine_dataset_no_outliers = handle_outliers_zscore(wine_dataset)

# Display the DataFrame after handling outliers
print(wine_dataset_no_outliers)
df = wine_dataset_no_outliers
####################################################################################
#Scale numerical feature to scale mean and SD 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler_standard = StandardScaler()
df_standardized = pd.DataFrame(scaler_standard.fit_transform(df), columns=df.columns)
df_standardized
#Here we have scaled the mean at 0 and standar deviation at 1
######################################################################3
# Normalize the data
scaler_minmax = MinMaxScaler()
df_normalized = pd.DataFrame(scaler_minmax.fit_transform(df), columns=df.columns)

# Display the DataFrame after normalization
df_normalized
#Scale numerical features to a specific range (between 0 and 1).
##################################################################################
#Convert categorical variables into binary vectors.
df_encoded = pd.get_dummies(df, columns=['Type'], prefix='Type')

# Display the DataFrame after One-Hot Encoding
print(df_encoded)
#################################################################################
#Use Log Transfrmation to stabilize variance and make the data more normally distributed.
df_log = np.log1p(df_encoded)  #np.log1p used to handle zero values

# Display the DataFrame after log transformation
print(df_log)

####################################################################################

X=df_log.values

#Let’s perform the K-Means clustering for n_clusters=3.
from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=3)
kmeans.fit(X)

kmeans.cluster_centers_
kmeans.labels_

#Minimize the dataset from 15 features to 2 features using principal component analysis (PCA).
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
 
reduced_X=pd.DataFrame(data=pca.fit_transform(X),columns=['PCA1','PCA2'])
 
#Reduced Features
reduced_X.head()

#Reducing the cluster centers using PCA.
centers=pca.transform(kmeans.cluster_centers_)
 
# reduced centers
centers

#Represent the cluster plot based on PCA1 and PCA2. 
#Differentiate clusters by passing a color parameter as c=kmeans.labels_
import matplotlib.pyplot as plt
plt.figure(figsize=(7,5))
 
# Scatter plot
plt.scatter(reduced_X['PCA1'],reduced_X['PCA2'],c=kmeans.labels_)
plt.scatter(centers[:,0],centers[:,1],marker='x',s=100,c='red')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('Wine Cluster')
plt.tight_layout()

pca.components_

