import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
import numpy as np
from sklearn.cluster import MeanShift
from sklearn import preprocessing
import pandas as pd 


"""
"""

df = pd.read_excel("dataset/titanic.xls")
original_df = pd.DataFrame.copy(df)
#print(df.head())

df.drop(["body","name"], 1, inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)
#print(df.head())

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
#print(df.head())

X = np.array(df.drop(["survived"], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df["survived"])

clf = MeanShift()
clf.fit(X)

label = clf.labels_
clusterCenters = clf.cluster_centers_

original_df["cluster_group"] = np.nan

#Referencing the row of the (titanic) dataframe and
#setting the value of whatever the value is in label
for i in range(len(X)):
    original_df["cluster_group"].iloc[i] = label[i]


nClusters_ = len(np.unique(label))

survivalRates = {}

for i in range(nClusters_):
    #Creating dataframe (temp_df) where the original dataframe 
    # is equal to a particular cluster group
    # eg (cluster 0:- when i is equal to 0)
    temp_df = original_df[ (original_df["cluster_group"] == float(i)) ]

    #Now the survivalCluster is a data that
    #describe where the people in temp_df survived
    survivalCluster = temp_df[(temp_df["survived"] == i)]

    #Calculate the survival rate of each cluster
    survivalRate = len(survivalCluster)/len(temp_df)

    #Populating our survivalRate
    survivalRates[i] = survivalRate

print(survivalRates)