import pandas as pd 
# import seaborn as sns 
import matplotlib.pyplot as plt 
import numpy as np 

dataset = pd.read_csv('FINAL_DATA.csv') 
dataset['Notification Open'] = dataset['Notification Open'].fillna(0)
dataset['Share Count'] = dataset['Share Count'].fillna(0)
dataset['Session Count'] = dataset['Session Count'].fillna(0)

from sklearn.model_selection import train_test_split
X = dataset.iloc[:, [7,10,11]].values
y = dataset.iloc[:, 12].values 
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.25, random_state = 109) 
from sklearn import svm
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))