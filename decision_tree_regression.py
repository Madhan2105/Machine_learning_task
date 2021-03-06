import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  #for plotting purpose
from sklearn import model_selection

dataset = pd.read_csv("FINAL_DATA.csv")
dataset['Notification Open'] = dataset['Notification Open'].fillna(0)
dataset['Share Count'] = dataset['Share Count'].fillna(0)
dataset['Session Count'] = dataset['Session Count'].fillna(0)   

print(dataset.head())
X = dataset.iloc[:, [7,10,11]].values
y = dataset.iloc[:, 13].values 

# import the regressor 
from sklearn.tree import DecisionTreeRegressor 

# create a regressor object 
regressor = DecisionTreeRegressor(random_state = 0) 

# fit the regressor with X and Y data 
regressor.fit(X, y) 

# predicting a new value 

# test the output by changing values, like 3750 
y_pred = regressor.predict([[6,190,192]])
# print the predicted price 
print("Predicted price: % d\n"% y_pred) 
scoring = 'neg_mean_absolute_error'
kfold = model_selection.KFold(n_splits=10, random_state=7, shuffle=True)
results = model_selection.cross_val_score(regressor, X, y, cv=kfold, scoring=scoring)
print("MAE: %.3f (%.3f)" % (results.mean(), results.std()))