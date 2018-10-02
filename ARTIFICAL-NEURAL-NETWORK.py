# -*- coding: utf-8 -*-
"""
______________ARTIFICIAL NEURAL NETWORK_______________________
1. INSTALL THEANO
2. INSTALL TENSORFLOW
3. INSTALL KERAS

@author : MD SAIF UDDIN

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values

#first we have to encode the state and gender class to dummy variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x_1 = LabelEncoder()
X[:, 1] = labelencoder_x_1.fit_transform(X[:, 1])
labelencoder_x_2 = LabelEncoder()
X[:, 2] = labelencoder_x_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1]) #converting three variable to two variable using dummy variable case
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]   #to avoiding the dummy variable trap

#splitting the data set into train data and test data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

#applying the featue scaling to the data to avoiding long calculation
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#importing the keras library
import keras
from keras.models import sequential
from keras.layers import Dense

#initiallising the ann
classifier = sequential()

#addding the input layer and hiddenlayer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11)) #relu stand for rectifier function 
#number of node in hiden layer is (N + 1)/2 : n is number of node in input layer and 1 is for one output layer , so we take average  off that.

#adding the second hiddenlayer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu')) #no needd to take input_dim in seccond layer itis compulsory for first layer

#adding the outputlayer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
#softmax to be used when there is more number of node in output layer

#compiling the ann
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#to find the optimal weight we have to apply stochastic gradeint descent which called adam
#binary_crossentropy is a loss function for binary output layer
#categorical underscore cross entropy is a loss function for multiple output layer

#fitting the ann to training set
classifier.fit(X_train, Y_train, batch_size = 10, nb_epoch = 100)

#predicting the test set results
Y_pred = classifier.predict(X_test)
Y_pred = (Y_pred > 0.5)

#making the confusion matrix
from sklearn.metrices import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)