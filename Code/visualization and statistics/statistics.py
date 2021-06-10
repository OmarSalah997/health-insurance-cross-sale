# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 18:45:49 2021

@author: somar
"""
import pandas as pd
data=pd.read_csv("train.csv")
pd.set_option("display.precision", 2)
cat=[]
num=[]
for i in data.columns:
    if data[i].dtype=="object":
           cat.append(i)        
    else:
            num.append(i)
         
data=data.drop('id', axis=1)
data.describe()
#convert categorial to numeric
data =pd.get_dummies(data, columns=cat,drop_first=True)  
corrMatrix = data.corr()
print(corrMatrix)