# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 23:20:50 2018

@author: MD SAIF UDDIN
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing  the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,4].values

#now convert country name to specific number .so, that it con be used in regression
#encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#avoiding dummy variable trap
X = X[:,1:] #not contain array 0

#splitting the data set into training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#fitting multiple linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()  #creating object
regressor.fit(X_train, Y_train)

#predicting the regression result
Y_pred = regressor.predict(X_test)

#building the optimal model using backward elimanation
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5 ]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()  #creating the object for statsmodel
regressor_OLS.summary()
X_opt = X[:, [0, 1, 3, 4, 5 ]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()  #creating the object for statsmodel
regressor_OLS.summary()
X_opt = X[:, [0, 3, 4, 5 ]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()#creating the object for statsmodel
regressor_OLS.summary()
X_opt = X[:, [0, 3, 5 ]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()#creating the object for statsmodel
regressor_OLS.summary()