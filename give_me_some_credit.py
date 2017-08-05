#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 12:42:35 2017

@author: johnnyhsieh
"""
import pandas as pd
import numpy as np

dataset = pd.read_csv('cs-training.csv',engine = 'python')
dataset.head()
y_train = dataset['SeriousDlqin2yrs']
#let see if the result is imbalance
y_train.value_counts()
#the result is 0 :139974 1:10026
x_train = dataset.drop(labels = ['SeriousDlqin2yrs','Unnamed: 0'],axis =1 )
#lets see which colume also empty
x_train.isnull().sum()
#we found out yes there's several row is miss MonthlyIncome 29731 
#and NumberOfDependents
x_train['MonthlyIncome'].median()
x_train['MonthlyIncome'].mode()
x_train['MonthlyIncome'].describe() 
x_train['NumberOfDependents'].describe()
x_train['NumberOfDependents'].mode()

#we fill up the missing row with mode value
fill_value = {'MonthlyIncome':int(x_train['MonthlyIncome'].mode())
,'NumberOfDependents': int(x_train['NumberOfDependents'].mode())}
x_train= x_train.fillna(fill_value)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x_train,y_train
                                                 ,test_size = 0.2
                                                 ,random_state = 42)

import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.callbacks import EarlyStopping
from sklearn.utils import class_weight

Classifier = Sequential()
#adam = optimizers.Adam(lr = 0.0005)
Classifier.add(Dense(units = 100, activation = 'relu'
                      ,kernel_initializer='uniform',input_dim = 10))
Classifier.add(Dropout(0.2))
Classifier.add(Dense(units = 10,activation = 'relu'
                     ,kernel_initializer = 'uniform'))
Classifier.add(Dense(units = 10,activation = 'relu'
                     ,kernel_initializer = 'uniform'))
Classifier.add(Dense(units = 10,activation = 'relu'
                     ,kernel_initializer = 'uniform'))
Classifier.add(Dense(units = 10,activation = 'relu'
                     ,kernel_initializer = 'uniform'))
Classifier.add(Dense(units = 5,activation = 'relu'
                     ,kernel_initializer = 'uniform'))
Classifier.add(Dense(1,activation = 'sigmoid',kernel_initializer = 'uniform'))
Classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy'
                   ,metrics = ['accuracy'])
weight = class_weight.compute_class_weight('balanced',np.unique(y_train)
,y_train)
early_stop = EarlyStopping(monitor = 'val_loss',patience = 1,mode = 'auto')
Classifier.fit(x_train,y_train,batch_size = 10,epochs = 30
               ,callbacks = [early_stop],validation_data = (x_test,y_test)
               ,shuffle = True, class_weight = weight)
score = Classifier.evaluate(x_test,y_test,batch_size = 10)
predict = Classifier.predict(x_test)
#I want to reduce the FN (false negative) because FN might
#case more loss for company
predict = (predict>0.4)

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn

cm = confusion_matrix((y_test),(predict))
plt.figure(figsize = (8,8))
plt.ylabel("True")
plt.xlabel("False")
sn.heatmap(cm,annot = True)
plt.show()







"""
the test data input
"""
#get to know the sumbit form
sample = pd.read_csv('sampleEntry.csv', engine ='python')
test_dataset = pd.read_csv('cs-test.csv', engine = 'python')
test_data = pd.read_csv('cs-test.csv', engine = 'python')
test_value_fill = {'MonthlyIncome':int(test_data['MonthlyIncome'].mode())
,'NumberOfDependents': int(test_data['NumberOfDependents'].mode())}
test_data = test_data.fillna(test_value_fill)
test_data = test_data.drop(labels = ['SeriousDlqin2yrs','Unnamed: 0'],axis = 1)
test_data = sc.fit_transform(test_data)
test_predict = Classifier.predict(test_data)

results = pd.DataFrame({
    'Id' : test_dataset['Unnamed: 0'],
    'Probability' : test_predict[:,0]
})
results.to_csv("reslut.csv",index = False)
