# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 13:08:52 2018

@author: MD SAIF UDDIN
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing data set into the training set and testset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

#fitting the dicison tree regressor to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, Y)

#we doesnot use feature scaling here because of trap

#predicting a new result
Y_pred = regressor.predict(6.5)

'''
#visualising the decision tree regressor results
plt.scatter(X, Y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('truth or Bulff(decision tree regressor)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
'''
# above  model is called non-contineous model
#so, we use high resolution and smoother curve
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('truth or Bulff(decision tree regressor)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

'''The result obbtain here is not so effective because of 2D plot,
   it is more powerful and effective in 3D
   But here it also predict very well
   '''
   
   '''
     THIS IS OF ONE TREE WHICH PREDICT VERY WELL BUT IF MULTIPLE TREE ARE
     PRESENT THEN IT IS GIVE MORE EFFICIENT REGRESSOR
     MULTIPLE TREE IS CALLED FOREST
     '''