# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 19:25:40 2021

@author: somar
"""

import numpy as np # linear algebra
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
train_data=pd.read_csv("train.csv")
#checking if there null values
train_data.isnull().sum()

#sns.countplot(train_data['Response'])
#sns.countplot(train_data['Driving_License'])

#sns.kdeplot(data=train_data, x="Region_Code", hue="Gender",fill=True, common_norm=False, palette="crest",alpha=.5, linewidth=0,)

#sns.kdeplot(data=train_data, x="Age", hue="Gender")
#sns.catplot(x="Vehicle_Damage", hue="Gender",kind="count", data=train_data)

#sns.catplot(x="Response", hue="Gender",kind="count", data=train_data)

#check vehicle age
#sns.countplot(train_data['Vehicle_Age'],palette="Set1")

#sns.distplot(x=train_data['Age'])


#age_damageYes=train_data[train_data['Vehicle_Damage'] =='Yes']

#sns.kdeplot( data=age_damageYes, x="Age", hue="Vehicle_Damage")

#sns.catplot(x="Response", hue="Age",kind="count", data=train_data)

#sns.countplot(data=train_data, x="Region_Code", hue="Vehicle_Damage")
#annual_res=train_data[train_data['Response'] == 1 ]
#g=sns.kdeplot(data=annual_res, x="Annual_Premium", hue="Response",fill=True, common_norm=False, palette="crest",alpha=.5, linewidth=0,)
#plt.show(g)
#g= (g.set(xlim=(0,100000)))
#sns.kdeplot(data=train_data, x="Policy_Sales_Channel", hue="Response",fill=True, common_norm=False)
#sns.kdeplot(data=train_data, x="Vintage", hue="Response")

#sns.catplot(x='Vehicle_Age', y='Annual_Premium', hue='Response', kind = 'bar', data = train_data)




#sns.catplot(x="Response", hue="Vehicle_Age",kind="count", data=train_data)
#sns.catplot(x="Response", hue="Previously_Insured",kind="count", data=train_data)

