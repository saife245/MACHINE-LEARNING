# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 21:59:06 2018

@author: MD SAIF UDDIN
"""
import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt

#importing the  data set
dataset = pd.read_csv("Market_Basket_Optimisation.csv", header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

#training apriori on data set
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

#visualising the results
results = list(rules)