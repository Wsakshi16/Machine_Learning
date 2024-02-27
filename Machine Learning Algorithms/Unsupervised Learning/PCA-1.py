# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 16:37:15 2023

@author: adity
"""



import numpy as np
import pandas as pd
uni1 = pd.read_excel("c:/0-datasets/University_Clustering.xlsx")
uni1.describe()

uni1.info()
uni = uni1.drop(["State"], axis=1)

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

#Considering only numerical data
uni.data = uni.iloc[:,1:]

#Normalizing the numerical data
uni_normal = scale(uni.data)
uni_normal

pca = PCA(n_components = 6)
pca_values = pca.fit_transform(uni_normal)

#The amount of variance that each PCA explains is
var = pca.explained_variance_ratio_
var

#PCA weights
#pca.components_
#pca.components_(0)

#Cumulative variance
var1 = np.cumsum(np.round(var, decimals = 4)* 100)
var1

#Variance plot for the PCA components obtained
plt.plot(var1, color = "red")

#PCA scores
pca_values

pca_data = pd.DataFrame(pca_values)
pca_data.columns = "comp0","comp1","comp2","comp3","comp4","comp5"
final = pd.concat([uni.Univ, pca_data.loc[:,0:3]],axis=1)
#This is 'Univ' column of uni data frame
#scatter diagram
import matplotlib.pyplot as plt
ax = final.plot(x="comp0", y ="comp1", kind="scatter", figsize=(12,8))
final[["comp0","comp1","Univ"]].apply(lambda x: ax.text(*x),axis =1 )
