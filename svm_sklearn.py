#Importing Libraries
import numpy as np
import pandas as pd
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split

#Loading our dataset
df = pd.read_csv("dataset/breast-cancer-wisconsin.data.txt")
df.drop(["id"], 1, inplace=True)
df.replace("?", -9999, inplace=True)

#Converting to numpy Array
X = np.array(df.drop(["class"],1))
y = np.array(df["class"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#Training our machine
clf = svm.SVC()
clf.fit(X_train, y_train)

accracy = clf.score(X_test, y_test)
print(accracy)

# Testing Testing
testSamples = clf.predict([[4,1,1,2,1,3,2,3,2],[4,2,1,3,4,3,1,2,2]])
print(testSamples)