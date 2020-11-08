import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  #for plotting purpose
from sklearn import linear_model   #for implementing multiple linear regression
from sklearn import model_selection

dataset = pd.read_csv("FINAL_DATA.csv")
dataset['Notification Open'] = dataset['Notification Open'].fillna(0)
dataset['Share Count'] = dataset['Share Count'].fillna(0)
dataset['Session Count'] = dataset['Session Count'].fillna(0)
# print(dataset)
print(dataset.head())
X = dataset.iloc[:, [7,10,11]].values
y = dataset.iloc[:, 13].values 
print(X)
reg = linear_model.LinearRegression()     #initiating linearregression
reg.fit(X,y)
predictedCO2 = reg.predict([[6,190,192]])
print(predictedCO2)
scoring = 'neg_mean_absolute_error'
kfold = model_selection.KFold(n_splits=10, random_state=7, shuffle=True)
results = model_selection.cross_val_score(reg, X, y, cv=kfold, scoring=scoring)
print("MAE: %.3f (%.3f)" % (results.mean(), results.std()))