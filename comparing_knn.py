import numpy as np
import pandas as pd
import random
import k_Nearest_Neighbour_Algo as myKNNAlgo

# Loading, cleaning and shuffling the dataset
df = pd.read_csv("dataset/breast-cancer-wisconsin.data.txt")
df.drop(["id"], 1, inplace=True)
df.replace("?", -9999, inplace=True)
fullList = df.astype(float).values.tolist()
random.shuffle(fullList)

# Spliting the data set
testSize = 0.2
trainSet = {2:[], 4:[]}
testSet = {2:[], 4:[]}
trainData = fullList[:-int(testSize*len(fullList))]
testData = fullList[-int(testSize*len(fullList)):]

for i in trainData:
    trainSet[i[-1]].append(i[:-1])

for i in testData:
    testSet[i[-1]].append(i[:-1])

#Training, Testing And Accuracy
correct = 0
total = 0

for group in testSet:
    for data in testSet[group]:
        vote = myKNNAlgo.kNearestNeighbor(trainSet, data, k=5)
        if group == vote:
            correct += 1
        total +=1

print("Accuracy is " + str(correct/total))

