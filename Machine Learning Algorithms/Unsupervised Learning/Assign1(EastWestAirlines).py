# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 21:14:00 2023

@author: adity
"""

#Assignment 1 on EastWestAirlines

import pandas as pd
import matplotlib.pyplot as plt
#how import file from data set and create a dataframe
air = pd.read_excel("c:/0-datasets/EastWestAirlines.xlsx")
a = air.describe()
#we know that there is scale difference among columns
#which we have to remove
#either by nomalisation and standardisation
#whenever there is mixed data apply normalisation
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
#Now apply this normalization function to air dataframe
#For all the rows and column from 1 until end
#since 0 th column is ID hence skipped
df_norm=norm_func(air.iloc[:,1:])
#you can check the dataframe which is saled
#between values from 0 to 1
#you can apply descirbe() function to ne dataframe
d = df_norm.describe()

#before you apply clustering,you need to plot dendrogram first
#Now to create dendrogram,we need to measure distance,
#we have to import linkage package
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
#linkage function gives us hierachical or aglomerative clustering
#ref the the help for linkage
z = linkage(df_norm,method="complete",metric="euclidean")
plt.figure(figsize=(15,8));
plt.title("Hierarchical Clustering dendrogram");
plt.xlabel("Index");
plt.ylabel("Distance")
#ref help of dendrogram
#sch.dendrogram(z)
sch.dendrogram(z,leaf_rotation=0,leaf_font_size=10)
plt.show()


#dendrogram()
#applying agglemerative clustering choosing 3 as cluster from dendogram
#whaterver has been displayed in dendrogram is not clustering 
#it is just showing number of possible clusters
from sklearn.cluster import AgglomerativeClustering
h_complete=AgglomerativeClustering(n_clusters=3,linkage="complete",affinity="euclidean").fit(df_norm)

#apply labels to the clusters h_complete.labels
cluster_labels=pd.Series(h_complete.labels_)

#Assign this Series to air dataframe t column and name the column
air['clust']=cluster_labels
#we want to relocate the column 12 to 0th position
air1=air.iloc[:,[12,1,2,3,4,5,6,7,8,9,10,11]]
#now check air1 dataframe
air1.iloc[:,2:].groupby(air.clust).mean()
#from the ouput cluster 2 has got higesht Top10
#lowest accept reatio ,best factly ratio and highest expenses
#highest geaduate ratio
air1.to_csv("New_EastWestAirlines.csv",encoding="utf-8")
import os
os.getcwd()

####################################################################################

































