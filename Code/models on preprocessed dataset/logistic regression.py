# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 20:19:49 2021

@author: somar
"""

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
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
# Scale only columns that have values greater than 1
to_scale = [col for col in data.columns if data[col].max() > 1]
mms = MinMaxScaler()
scaled = mms.fit_transform(data[to_scale])
scaled = pd.DataFrame(scaled, columns=to_scale)

# Replace original columns with scaled ones
for col in scaled:
    data[col] = scaled[col]
   
X = data.drop('Response', axis=1)
X = X.drop('id', axis=1)

y = data['Response']  
sm = SMOTE(random_state=42)

X_sm, y_sm = sm.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.25 , random_state=1)
Log_reg = LogisticRegression(solver='liblinear', random_state=0)
Log_reg.fit(X_train, y_train)
Log_reg.predict(X_test)



Y_trainset_prediction = Log_reg.predict(X_train)
model_score_onTrainingSet = Log_reg.score(X_train, y_train)
print(model_score_onTrainingSet)
print(metrics.confusion_matrix(y_train, Y_trainset_prediction))
print(metrics.classification_report(y_train, Y_trainset_prediction))


y_testSet_prediction = Log_reg.predict(X_test)
model_score_onTestgSet = Log_reg.score(X_test, y_test)
print(model_score_onTestgSet)
print(metrics.confusion_matrix(y_test, y_testSet_prediction))
print(metrics.classification_report(y_test, y_testSet_prediction))
