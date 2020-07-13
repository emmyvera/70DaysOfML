import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
import warnings
from collections import Counter

#Data
#dataset = {"g":[[2,3],[1,3],[2,1]], "r":[[5,6],[7,5],[6,7]]}
#new_features = [6,5]

#Trying to graph our work
# [[plt.scatter(li[0],li[1],s=100,c=i) for li in dataset[i]] for i in dataset]
# plt.scatter(new_features[0], new_features[1], s=100, c="b")
# plt.show()

def kNearestNeighbor(data, predict, k=3):
    if len(data)>= k:
        warnings.warn("K is set to a value less than the total voting group!")

    distance = []
    for group in data:
        for features in data[group]:
            eulideanDistance = np.linalg.norm(np.array(features) - np.array(predict))
            distance.append([eulideanDistance, group])

    vote = [i[1] for i in sorted(distance)[:k]]
    voteResult = Counter(vote).most_common(1)[0][0]

    return voteResult

#result = kNearestNeighbor(dataset, new_features, k=5)

#Trying to graph our work
# [[plt.scatter(li[0],li[1],s=100,c=i) for li in dataset[i]] for i in dataset]
# plt.scatter(new_features[0], new_features[1], s=100, c=result)
# plt.show()