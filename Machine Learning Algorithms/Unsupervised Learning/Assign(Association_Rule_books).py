# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 16:34:03 2023

@author: aditya
"""

'''
Business Objective:
   
maximize: Identify and promote book combinations that are 
frequently purchased together to increase cross-selling opportunities.
 
Minimize: Increase sales and revenue by promoting popular book categories 
  
Constraints: The business needs to address online competition.
Strategies should include both online and offline components to capture a broader market.
'''

#data dictionary

# Nominal Data:

# 'ChildBks': Children's books category.
# 'YouthBks': Youth books category.
# 'CookBks': Cookbooks category.
# 'RefBks': Reference books category.
# 'ArtBks': Art books category.
# 'GeogBks': Geography books category.
# 'ItalCook': Italian Cookbooks category.
# 'ItalAtlas': Italian Atlases category.
# 'ItalArt': Italian Art books category.
# 'Florence': Possibly a location or specific book related to Florence.


# Ordinal Data:
# 'DoItYBks': Do-it-yourself books category.
 
#data cleaning 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

book=pd.read_csv('book.csv')
book

####################################
book.columns
'''
['ChildBks', 'YouthBks', 'CookBks', 'DoItYBks', 'RefBks', 'ArtBks',
       'GeogBks', 'ItalCook', 'ItalAtlas', 'ItalArt', 'Florence']
'''

book.shape
#(2000, 11)

book.dtypes
# all featues have numerical values

a=pd.isnull(book)
a.sum()
# there no null value in the dataset

q=book.value_counts()
q

v=book.describe()
# we get 5 number sumarry
# The mean value is near to zero and also the standard deviation is 
# near to zero and the meadian value for the all datapoints is zero
book.info()
# This will give us the informationn about all the points

##############################################################################

plt.figure(figsize=(12,8))
sns.heatmap(book.isnull(),cmap='viridis');

book.value_counts()

book.loc[:,:].sum()

#we get number ok books in each feature

for i in book.columns:
    print(i)
    print(book[i].value_counts())
    print()


# we get information here that in each feature and id how many books are there

### now we will cheak supporting values
# Product Frequency / Total Sales

first = pd.DataFrame(book.sum() / book.shape[0], columns = ["Support"]).sort_values("Support", ascending = False)
first

'''
           Support
CookBks     0.4310
ChildBks    0.4230
DoItYBks    0.2820
GeogBks     0.2760
YouthBks    0.2475
ArtBks      0.2410
RefBks      0.2145
ItalCook    0.1135
Florence    0.1085
ItalArt     0.0485
ItalAtlas   0.0370
'''
# from this we can understand which feature contributing how much

first[first.Support >= 0.1]

first[first.Support >= 0.15]
first[first.Support >= 0.20]

'''
          Support
CookBks    0.4310
ChildBks   0.4230
DoItYBks   0.2820
GeogBks    0.2760
YouthBks   0.2475
ArtBks     0.2410
RefBks     0.2145
'''
# we have consider less than 0.1 ,0.15,0.2 are not so imp for model building

# we will check by associating 2,3,4 values

import itertools
second = list(itertools.combinations(first.index, 2))
second = [list(i) for i in second]
second[:10]

# Finding support values
value = []
for i in range(0, len(second)):
    temp = book.T.loc[second[i]].sum() 
    temp = len(temp[temp == book.T.loc[second[i]].shape[0]]) / book.shape[0]
    value.append(temp)
# Create a data frame            
secondIteration = pd.DataFrame(value, columns = ["Support"])
secondIteration["index"] = [tuple(i) for i in second]
secondIteration['length'] = secondIteration['index'].apply(lambda x:len(x))
secondIteration = secondIteration.set_index("index").sort_values("Support", ascending = False)
# Elimination by Support Value
secondIteration = secondIteration[secondIteration.Support > 0.1]
secondIteration

# now will try for 3
second = list(itertools.combinations(first.index, 3))
second = [list(i) for i in second]


# Finding support values

value = []
for i in range(0, len(second)):
    temp = book.T.loc[second[i]].sum() 
    temp = len(temp[temp == book.T.loc[second[i]].shape[0]]) / book.shape[0]
    value.append(temp)
# Create a data frame            
secondIteration = pd.DataFrame(value, columns = ["Support"])
secondIteration["index"] = [tuple(i) for i in second]
secondIteration['length'] = secondIteration['index'].apply(lambda x:len(x))
secondIteration = secondIteration.set_index("index").sort_values("Support", ascending = False)
# Elimination by Support Value
secondIteration = secondIteration[secondIteration.Support > 0.1]
secondIteration


# now we will try for 4
second = list(itertools.combinations(first.index, 4))
second = [list(i) for i in second]
# Sample of combinations
second[:10]



# Finding support values
value = []
for i in range(0, len(second)):
    temp = book.T.loc[second[i]].sum() 
    temp = len(temp[temp == book.T.loc[second[i]].shape[0]]) / book.shape[0]
    value.append(temp)
# Create a data frame            
secondIteration = pd.DataFrame(value, columns = ["Support"])
secondIteration["index"] = [tuple(i) for i in second]
secondIteration['length'] = secondIteration['index'].apply(lambda x:len(x))
secondIteration = secondIteration.set_index("index").sort_values("Support", ascending = False)
# Elimination by Support Value
secondIteration = secondIteration[secondIteration.Support > 0.1]
secondIteration

## Association Rules
from mlxtend.frequent_patterns import apriori,association_rules
## Association rules with 10% Support and 30% confidence

# With 10% Support
frequent_itemsets=apriori(book,min_support=0.1,use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
frequent_itemsets

# with 30% Confidence
rules = association_rules(frequent_itemsets,metric='confidence', min_threshold=0.3)
rules

rules.sort_values('lift',ascending=False)

lift=rules[rules.lift>1]
lift

# visualization of obtained rule
plt.figure(figsize=(16,9))
plt.scatter(rules['support'],rules['confidence'])
plt.xlabel('support')
plt.ylabel('confidence') 
plt.show()

matrix = lift.pivot('antecedents','consequents','lift')
matrix

plt.figure(figsize=(20,6),dpi=250)
sns.heatmap(matrix,annot=True)
plt.title('HeatMap - ForLiftMatrix')
plt.yticks(rotation=0)
plt.xticks(rotation=90);

plt.figure(figsize=(20,6),dpi=250)
sns.barplot("support","confidence",data=lift)
plt.title("support vs confidence")
plt.show()

plt.figure(figsize=(16,9))
fit = np.polyfit(rules['lift'], rules['confidence'], 1)
fit_fn = np.poly1d(fit)
plt.plot(rules['lift'], rules['confidence'], 'yo', rules['lift'], 
fit_fn(rules['lift']))
plt.xlabel('lift')
plt.ylabel('Confidence')
plt.title('lift vs Confidence')

# With 15% Support
frequent_itemsets=apriori(book,min_support=0.15,use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
frequent_itemsets

# with 40% Confidence
rules = association_rules(frequent_itemsets,metric='confidence', min_threshold=0.4)
rules

rules.sort_values('lift',ascending=False)

# Lift Ratio > 1 is a good influential rule in selecting the associated transactions
lift=rules[rules.lift>1]
lift
# visualization of obtained rule
plt.figure(figsize=(16,9))
plt.scatter(rules['support'],rules['confidence'])
plt.xlabel('support')
plt.ylabel('confidence') 
plt.show()


matrix = lift.pivot('antecedents','consequents','lift')
matrix

plt.figure(figsize=(20,6),dpi=250)
sns.heatmap(matrix,annot=True)
plt.title('HeatMap - ForLiftMatrix')
plt.yticks(rotation=0)
plt.xticks(rotation=90);

plt.figure(figsize=(20,6),dpi=250)
sns.barplot("support","confidence",data=lift)
plt.title("support vs confidence")
plt.show()

plt.figure(figsize=(16,9))
fit = np.polyfit(rules['lift'], rules['confidence'], 1)
fit_fn = np.poly1d(fit)
plt.plot(rules['lift'], rules['confidence'], 'yo', rules['lift'], 
fit_fn(rules['lift']))
plt.xlabel('lift')
plt.ylabel('Confidence')
plt.title('lift vs Confidence');

## Association rules with 20% Support and 60% confidence
# With 20% Support
frequent_itemsets=apriori(book,min_support=0.2,use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
frequent_itemsets

# with 60% Confidence
rules = association_rules(frequent_itemsets,metric='confidence', min_threshold=0.6)
rules

rules.sort_values('lift',ascending=False)

 # Lift Ratio > 1 is a good influential rule in selecting the associated transactions
lift=rules[rules.lift>1]
lift

# visualization of obtained rule
plt.figure(figsize=(16,9))
plt.scatter(rules['support'],rules['confidence'])
plt.xlabel('support')
plt.ylabel('confidence') 
plt.show()

matrix = lift.pivot('antecedents','consequents','lift')
matrix

plt.figure(figsize=(20,6),dpi=250)
sns.heatmap(matrix,annot=True)
plt.title('HeatMap - ForLiftMatrix')
plt.yticks(rotation=0)
plt.xticks(rotation=90)

plt.figure(figsize=(20,6),dpi=250)
sns.barplot("support","confidence",data=lift)
plt.title("support vs confidence")
plt.show()







































# =============================================================================
# # step 4
# =============================================================================



sns.boxplot(df,x='ChildBks')
# No outlier 
sns.boxplot(df,x='YouthBks')
#There is one outlier 
sns.boxplot(df,x='CookBks')
# No Outlier
sns.boxplot(df,x='RefBks')
# There is one outlier
sns.boxplot(df)
# Observe that some columns contain  the outlier so we have to normalize it

#Normalization
#The data is numeric one so we have to perform normalization

def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

df_norm=norm_fun(v)
df_norm

# Pairplot
sns.pairplot(df)
# No Datapoints are corelated as the all the datapoints are in scatter form 

# Heatmap
corr=df.corr()
sns.heatmap(corr)
# The diagonal color of the heatmap is same as the datapoints folllow some pattern
# so we can use this data for the model building
############################################

b=df_norm.describe()

sns.boxplot(df_norm)
# No Outlier is remaining
# The all the quantile points are converted in the rande of 0-1

# =============================================================================
# # step 5 Model Building
# =============================================================================

# Association Rules
from mlxtend.frequent_patterns import apriori,association_rules

data=pd.read_csv('book.csv')

data

# All the data is in properly separated form so no need to apply the encoding techique
# as it is already is in the form of numeric one

from collections import Counter
item_frequencies=Counter(data)

# Apriori algorithm
frequent_itemsets = apriori(data, min_support=0.05, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# This generate association rule for columns
# comprises of antescends,consequences

rules.head(20)
rules.sort_values('lift',ascending=False).head(10)

# Visualize the rules
import networkx as nx
import matplotlib.pyplot as plt



# Create directed graph from the rules
G = nx.from_pandas_edgelist(rules, 'antecedents', 'consequents')

# Draw the graph
fig, ax = plt.subplots(figsize=(14, 8))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=2500, node_color="skyblue", font_size=10, font_color="black", font_weight="bold", edge_color="gray", linewidths=1, alpha=0.7)
plt.title("Association Rules Network", fontsize=15)
plt.show()

# the benefits/impact of the solution 
# By identifying books that are frequently purchased together,
# the bookstore can create curated bundles or recommendations, enhancing the overall 
# shopping experience for customers.
# By using this association rule we can stratergically placed the books together to encourage
# the customer to purchased more items which will help to increased the overall revenue