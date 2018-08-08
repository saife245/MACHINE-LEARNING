import numpy as np# library contain math libraries
import pandas as pd #import data set and manage them with help of this lib
import matplotlib.pyplot as plt #it is used to plot graph

#importing dataset with the help of panda
dataset = pd.read_csv('Data.csv')
Y = dataset.iloc[:,3].values #iloc[] is used to select the precise location of list
X = dataset.iloc[:,:-1].values #we take all the line of dataset except last line

#taking care of missing data
from sklearn.preprocessing import Imputer #sklearn library is used to make michenary model
#create the object
imputer = Imputer(missing_values = "NaN",strategy = 'mean',axis = 0)
imputer.fit(X[:,1:3])#we choose upper bound i.e.3in python not  we choose 2
X[:,1:3] = imputer.transform(X[:,1:3])

#encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()#making object and encoded the name of country
X[:,0] = labelencoder_X.fit_transform(X[:,0]) #wwe replace the country namee with encded number so put in column 1
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()#making object and encoded the name of country
Y = labelencoder_Y.fit_transform(Y)

#splitting the dataset into training set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)#test size would we choose in wich devide the dataset into trainingset testset
#for betterment we choose test setto be 20 or 25%


#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
