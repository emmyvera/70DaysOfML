import numpy as np
import pandas as pd
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split

df = pd.read_csv()
df.drop(["id"], 1, inplace=True)

X = np.array()
y = np.array()

X_train, X_test, y_train, y_test = train_test_split()

clf = neighbor.KNearestNeighbor()
clf.fit(X_train, y_train)

accracy = clf.score(X_test, y_test)
print(accracy)