import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

col_names = ['User ID', 'Notification Open', 'Session Count', 'Share Count', 'Uninstalled']
# load dataset
dataset = pd.read_csv("FINAL_DATA.csv")
dataset['Notification Open'] = dataset['Notification Open'].fillna(0)
dataset['Share Count'] = dataset['Share Count'].fillna(0)
dataset['Session Count'] = dataset['Session Count'].fillna(0)

print(dataset.head())
X = dataset.iloc[:, [7,10,11]].values
y = dataset.iloc[:, 12].values 
print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) 
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))