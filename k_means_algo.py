import matplotlib.pyplot as plt 
import numpy as np 
from matplotlib import style
style.use("ggplot")


X = np.array([[1,2],
            [1.5,3],
            [6,8],
            [8,8],
            [1,0],
            [9,11]])

plt.scatter(X[:,0], X[:,1], s=100)
plt.show()

colors = 10*["g","r","c","b","k","o"]

class KMeans:
    def __init__(self, k=2, tol=0.001, maxIter=300):
        self.k = k
        self.tol = tol
        self.maxIter = maxIter

    def fit(self, data):
        self.centroids = {}

        #Picking Our First Centroids
        for i in range(self.k):
            self.centroids[i]=data[i]

        #Keeping the Centroids & Classification
        for i in range(self.maxIter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for featureset in data:
                #Here we are calculating the distance between a our featureset and the centroids
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            optimized = True

            for c in self.centroids:
                originalCentroid = prev_centroids[c]
                currentCentroid = self.centroids[c]

                if np.sum((currentCentroid-originalCentroid)/ originalCentroid *100)> self.tol:
                    optimized = False

            if optimized:
                break


    def predict(self, data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification



clf = KMeans()
clf.fit(X)

for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],
    marker="o", color="k", s=150, linewidths=5)


for classification in clf.classifications:
    color = colors[classification]
    print(color)

    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker="x",
        s=150, color=color, linewidths=5)

plt.show()