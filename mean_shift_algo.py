import matplotlib.pyplot as plt 
import numpy as np 
from matplotlib import style
style.use("ggplot")


X = np.array([[1,2],
            [1.5,3],
            [6,8],
            [8,8],
            [1,0.6],
            [9,11],
            [8,2],
            [10,2],
            [9,3]])

plt.scatter(X[:,0], X[:,1], s=100)
plt.show()

colors = 10*["g","r","c","b","k","o"]

class MeanShift:
    def __init__(self, radius=4):
        self.radius = radius


    def fit(self, data):
        centroids = {}

        for i in range(len(data)):
            centroids[i] = data[i]

        while True:
            newCentroids = []
            for i in centroids:
                inBandwidth = []
                centroid = centroids[i]
                for featureset in data:
                    if np.linalg.norm(featureset-centroid) < self.radius:
                        inBandwidth.append(featureset)
                newCentroid = np.average(inBandwidth,axis=0)
                newCentroids.append(tuple(newCentroid))

            uniques = sorted(list(set(newCentroids)))

            prevCentroids = dict(centroids)

            centroids = {}
            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])

            optimized = True

            for i in centroids:
                if not np.array_equal(centroids[i], prevCentroids[i]):
                    optimized = False

                if not optimized:
                    break

            if  optimized:
                break

        self.centroids = centroids

    def predict(self, data):
        pass


clf = MeanShift()
clf.fit(X)

centroids = clf.centroids

plt.scatter(X[:,0], X[:,1], s=150)

for c in centroids:
    plt.scatter(centroids[c][0], centroids[c][1], c="k", marker="*", s=150)

plt.show()