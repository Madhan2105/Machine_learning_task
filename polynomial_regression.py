# Importing the libraries 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn import model_selection

dataset = pd.read_csv("FINAL_DATA.csv")
dataset['Notification Open'] = dataset['Notification Open'].fillna(0)
dataset['Share Count'] = dataset['Share Count'].fillna(0)
dataset['Session Count'] = dataset['Session Count'].fillna(0)
print(dataset.head())
X = dataset.iloc[:, [7,10,11]].values
y = dataset.iloc[:, 13].values 
# Fitting Linear Regression to the dataset 
from sklearn.linear_model import LinearRegression 
lin = LinearRegression() 

lin.fit(X, y) 

# Fitting Polynomial Regression to the dataset 
from sklearn.preprocessing import PolynomialFeatures 

poly = PolynomialFeatures(degree = 4) 
X_poly = poly.fit_transform(X) 

poly.fit(X_poly, y) 
lin2 = LinearRegression() 
lin2.fit(X_poly, y) 

# Predicting a new result with Linear Regression 
predicted = lin.predict([[6,190,192]]) 
print(predicted)
scoring = 'neg_mean_absolute_error'
kfold = model_selection.KFold(n_splits=10, random_state=7, shuffle=True)
results = model_selection.cross_val_score(lin, X, y, cv=kfold, scoring=scoring)
print("MAE: %.3f (%.3f)" % (results.mean(), results.std()))