# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 17:24:25 2023

@author: aditya
"""
pip install mlxtend
from mlxtend.frequent_patterns import apriori,assocoation_rules
#Here we are going to use transactional data wherein size of each row is not 
#we can not use  pandas to load this unstructured data
#here function called open() is used
#create an empty list
groceries=[]
with open("c:/0-datasets/groceries.csv") as f:groceries=f.read()
#splitting the data into seperate trancsactions using seperator, it is comma seperator
#We can use new line character "\n"
groceries = groceries.split("\n")
#Earlier groceries datastructure was in string format, now it will change to 
#9836, each item is comma seperated 
#our main aim is to calculate #A, #C,
#we will have to seperate out each item from each transaction 
groceries_list=[]
for i in groceries:
    f=groceries_list.append(i.split(","))
#split function will seperate each item from each list, wherever it will find 
#in order tp generate association rules, you can directly use groceries_list  
#Now let us seperate out each item from the groceries list
all_groceries_list=[ i for item in groceries_list for i in item]
#You will get all the items occured in all the transactions
#We will get 43368 items in various transactions

#Now let us count the frequency of each item
#we will import collections package which has counter function which will count 
from collections import Counter
item_frequencies=Counter(all_groceries_list) 
#item_frequencies is basically dictionary having x[0] as key and x[1] as value
#we want to access values and sort based on the count that occured in it.
#It will show the count of each item purchased in every transactions 
#now let us sort these frequencies in ascending order
item_frequencies=sorted(item_frequencies.items(),key=lambda x:x[1])
#when we execute this, item frequencies will be in sorted form,
#in ths form of tuple 
#item name with count
#Let us seperate out items and their count
items=list(reversed([i[0] for i in item_frequencies]))
#This is list comprehension for each item in item frequencies access the key
#Here you will get items list
frequencies=list(reversed([i[1] for i in item_frequencies]))
#here you will get count purchase of each item

#Now let us plot bar graph of each item frequencies
import matplotlib.pyplot as plt
#here we are taking frequencies from 0 to 11, you can try 0-15 or any other
plt.bar(height=frequencies[0:11],x=list(range(0,11)))
plt.xticks(list(range(0,11)),items[0:11])
##plt.xticks, You can specify a rotation for the tick 
#labels in degrees or with keywords.
plt.xlabel("items")
plt.ylabel("count")
plt.show()
import pandas as pd
#Now let us try to establish association rule mining 
#we have groceries list in the list format, we need 
#to convert it in dataframe format
groceries _series=pd.DataFrame(pd.Series(groceries_list))
#Now we will bget dataframa of size 9836X1 size, column
#comprises of multiple items 
#we had extra row created, check the groceries_series,
#last row is empty, let us first delete it
groceries_series = groceries_series.iloc[:9835,:]
#We have taken rows from 0 to 3834 and column 0 to all
#groceries series has column having name 0, let us rename as transactions
grocereis_series.columns=["Transactios"]
#Now we will have to apply 1-hot encoding, before that in 
#one column ther are various items seperated by ','
#let is seperate it with '*'
x=groceries_series["Transactions"].str.join(sep='*')
#check the x in variable explorer which has * seperator rather the ','
x=x.str.get_dummies(sep='*')
#you will get 1-hot encoded dataframe of size 9835X169
#This is our input data that apply to apriori algorithm,
#it will generate 169! rules, min support value
#is 0.0075(it must be between 0 to 1),
#you can give any number but must be between 0 and 1
frequent _itemsets=apriori(x,min_support=0.0075,max_len=4,use_colnames=True)
#you will get support values from 1,2,3 and 4 max items
#let us sort these support values
frequent_itemsets.sort_values('support',ascending=False,inplace=True)
#Support values will be sorted in descending order 
#Even EDA was also have the same trend, in EDA there was count
#and there it is support value
#we will generate association rules, this association
#rule will calculate all the matrix
#of each and every combination
rules=association_rules(frequent_itemsets,metrices="lift",)






