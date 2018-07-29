# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 22:44:10 2018

@author: SAIF UDDIN
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values

#slitting the dataset into the training set
from sklearn.cross_validation import train_test_split
X_train,X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2 , random_state = 0)

#fitting simple inear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#predicting the test set result
Y_pred = regressor.predict(X_test)

#visualising the training set result
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('salary vs Experiance (Training set)')
plt.xlabel('years of Experiance')
plt.ylabel('salary')
plt.show()

#visualising the test set result
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')#we not have to change train to test because it gave same result
plt.title('salary vs Experiance (Test set)')
plt.xlabel('years of Experiance')
plt.ylabel('salary')
plt.show()