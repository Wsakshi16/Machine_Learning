# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 15:28:25 2023

@author: adity
"""
'''
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
'''


import numpy as np
import pandas as pd
transaction_data = pd.read_csv("c:/0-datasets/transactions_retail.csv")
transaction_data.describe()


























