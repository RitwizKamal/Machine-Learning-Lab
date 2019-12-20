# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 11:27:21 2019

@author: Ritwiz
"""

#Loading the dataset
import pandas as pd
import numpy as np
dff=pd.read_csv('C:/Users/Ritwiz/OneDrive/Desktop/Project/BHU/2/feature selection/breast_cancer_wisconsin_diagnosis.csv')
df=dff.values
df_copy=dff.values


#from sklearn.model_selection import train_test_split  
#X_train, X_test, y_train, y_test = train_test_split(data1, class_labels1, test_size = 0.20)

#Naive-Bayes Classifier
v_acc=[]
print("NAIVE-BAYES CLASSIFIER:")
import random as r
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,classification_report
clf = GaussianNB()
from sklearn.model_selection import cross_val_score
for j in range(5):
    r.shuffle(df_copy)
    data=df_copy[:,2:32]
    class_labels=df_copy[:,1]
    scores=cross_val_score(clf,data,class_labels,cv=5)
    for i in range(scores.shape[0]):
        v_acc.append(scores[i])
print("Mean Validation Accuracy:",sum(v_acc)/len(v_acc))

#clf.fit(X_train, y_train)
#y_pred=clf.predict(X_test)
#accuracy=accuracy_score(y_test,y_pred)
#print("Test Accuracy=",accuracy)
#print(classification_report(y_test,y_pred))


#-----------------------------------------------------------------------------------------

#Logistic Regression
from sklearn.linear_model import LogisticRegression
clf1 = LogisticRegression(random_state=0, solver='liblinear',multi_class='ovr')
print("\n")
v_acc2=[]
print("LOGISTIC REGRESSION:")
for j in range(5):
    r.shuffle(df)
    data2=df[:,2:32]
    class_labels2=df[:,1]
    scores2=cross_val_score(clf1,data2,class_labels2,cv=5)
    for i in range(scores2.shape[0]):
        v_acc2.append(scores2[i])
print("Mean Validation Accuracy:",sum(v_acc2)/len(v_acc2))

#clf1.fit(X_train, y_train)
#y_pred1 = clf1.predict(X_test)
#accuracy1=accuracy_score(y_test,y_pred1)
#print("Accuracy=",accuracy1)
#print(classification_report(y_test,y_pred1))


