# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 14:08:09 2020

@author: bibek
"""
import numpy as np
from sklearn import preprocessing,model_selection,neighbors
import pandas as pd
import pickle

df = pd.read_csv('breast_cancer_csv.data.txt')
df.replace('?',-99999 ,inplace=True) # datset consits ? for missing data so replace
df.drop(['id'],1,inplace=True) # remove id field which is not necessary for prediction #impact accuracy

X = np.array(df.drop(['class'],1)) #features

y = np.array(df['class']) #label or class

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2)

clf = neighbors.KNeighborsClassifier()
#clf.fit(X_train, y_train) #trainng once for creating pickel

#writing in file

#pickle_wb = open('model.pickle','wb')
#pickle.dump(clf,pickle_wb)

pickle_rb = open('model.pickle','rb')
clf = pickle.load(pickle_rb)
accuracy = clf.score(X_test,y_test)

print(accuracy)

example_measures = np.array([[4,2,1,1,1,2,3,2,1]]) #without id and class
#example_measures = example_measures.reshape(1,-1) # because we dont have id and class so we need to reshape # remove deprecate warning

#example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,1,1,2,3,2,1]]) 
example_measures = example_measures.reshape(len(example_measures),-1)

prediction = clf.predict(example_measures)

print(prediction)





