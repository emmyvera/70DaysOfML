import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd 


"""
"""

df = pd.read_excel("dataset/titanic.xls")
#print(df.head())

df.drop(["body","name"], 1, inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)
print(df.head())

def handleCategoricalData(df):
    columns = df.columns.values

    for column in columns:
        textDigitVal = {}
        
        def convertToIntVal(val):
            return textDigitVal[val]
        
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x=0
            for unique in unique_elements:
                if unique not in textDigitVal:
                    textDigitVal[unique] = x
                    x+=1

            df[column] = list(map(convertToIntVal, df[column]))
    return df

df = handleCategoricalData(df)
print(df.head())

X = np.array(df.drop(["survived"], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df["survived"])

clf = KMeans(n_clusters=2)
clf.fit(X)

correct = 0 
for i in range(len(X)):
    predictThis = np.array(X[i].astype(float))
    predictThis = predictThis.reshape(-1, len(predictThis))
    prediction = clf.predict(predictThis)
    if prediction[0] == y[1]:
        correct += 1

print(correct/len(X))
