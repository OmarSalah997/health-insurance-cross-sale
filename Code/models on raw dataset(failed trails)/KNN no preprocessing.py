# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 19:23:08 2021

@author: somar
"""
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
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
KNN=KNeighborsClassifier()
KNN.fit(X_train,y_train)
Y_trainset_prediction = KNN.predict(X_train)
model_score_onTrainingSet = KNN.score(X_train, y_train)
print(model_score_onTrainingSet)
print(metrics.confusion_matrix(y_train, Y_trainset_prediction))
print(metrics.classification_report(y_train, Y_trainset_prediction))


y_testSet_prediction = KNN.predict(X_test)
model_score_onTestgSet = KNN.score(X_test, y_test)
print(model_score_onTestgSet)
print(metrics.confusion_matrix(y_test, y_testSet_prediction))
print(metrics.classification_report(y_test, y_testSet_prediction))




