# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 23:06:29 2021

@author: somar
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 19:13:43 2021

@author: somar
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE 
data=pd.read_csv("train.csv")
cat=[]
num=[]
for i in data.columns:
    if data[i].dtype=="object":
           cat.append(i)
            
    else:
            num.append(i)
#convert categorial to numeric              
data =pd.get_dummies(data, columns=cat,drop_first=True)  
X = data.drop('Response', axis=1)
X = X.drop('id', axis=1)

y = data['Response']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25 , random_state=1)
            
RF_model=RandomForestClassifier(n_estimators=100,random_state=1)
RF_model.fit(X_train, y_train)
y_train_predict = RF_model.predict(X_train)
model_score =RF_model.score(X_train, y_train)
print(model_score)
print(metrics.confusion_matrix(y_train, y_train_predict))
print(metrics.classification_report(y_train, y_train_predict))
y_test_predict = RF_model.predict(X_test)
model_scoreRF = RF_model.score(X_test, y_test)
print(model_scoreRF)
print(metrics.confusion_matrix(y_test, y_test_predict))
print(metrics.classification_report(y_test, y_test_predict))