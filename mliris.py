# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 11:12:55 2017

@author: adityan
"""

import pandas as pd
import sklearn

#Read the data from the csv file
f = open("iris.txt")
f.readline()  # skip the header
data = pd.read_csv(f)


#Divide to training and testing data sets

#Getting features except the last column i.e the label
x_features = data.iloc[25:125, :-1]
#Getting label for the exracted features i.e only last column
x_label = data.iloc[25:125, -1]

#Testing data
#Splitting it in weird way into 3 dataframes
x1_test_features = data.iloc[100:125, :-1]
x1_test_label = data.iloc[100:125, -1]

#you can decide how to split data
x2_test_features = data.iloc[0:-1, :-1]
x2_test_label = data.iloc[0:-1, -1]


x3_test_features = data.iloc[50:75, :-1]
x3_test_label = data.iloc[50:75, -1]


#Conactenating the split data to test
x_test_features = [x1_test_features,x2_test_features,x3_test_features]
x_test_f = pd.concat(x_test_features)

x_test_label = [x1_test_label,x2_test_label,x3_test_label]
x_test_l = pd.concat(x_test_label)


#decision tree classifier
#you can use any other classifier as Randomforest,Knn etc
clf = tree.DecisionTreeClassifier()
clf= clf.fit(x_features, x_label)
predictions = clf.predict(x_test_f)
print(predictions)

#accuracy definition
from sklearn.metrics import accuracy_score
print("Accuracy:",accuracy_score(x_test_l, predictions) * 100)