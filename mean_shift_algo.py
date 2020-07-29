import matplotlib.pyplot as plt 
import numpy as np 
from matplotlib import style
from sklearn.datasets.samples_generator import make_blobs
style.use("ggplot")

X, y = make_blobs(n_samples=15, centers=3, n_features=2)

# X = np.array([[1,2],
#             [1.5,3],
#             [6,8],
#             [8,8],
#             [1,0.6],
#             [9,11],
#             [8,2],
#             [10,2],
#             [9,3]])

# plt.scatter(X[:,0], X[:,1], s=100)
# plt.show()

colors = 10*["g","r","c","b","k","o"]

class MeanShift:
    def __init__(self, radius=None, radiusNormStep = 100):
        self.radius = radius
        self.radiusNormStep = radiusNormStep


    def fit(self, data):

        if self.radius == None:
            allDataCentroid = np.average(data, axis=0)
            allDataNorm = np.linalg.norm(allDataCentroid)
            self.radius = allDataNorm / self.radiusNormStep


        centroids = {}

        for i in range(len(data)):
            centroids[i] = data[i]

        weights = [i for i in range(self.radiusNormStep)][::-1]

        while True:
            newCentroids = []
            for i in centroids:
                inBandwidth = []
                centroid = centroids[i]



                for featureset in data:

                    distance = np.linalg.norm(featureset-centroid)
                    if distance == 0:
                        distance = 0.00000000001
                    weightsIndex = int(distance/self.radius)
                    if weightsIndex > self.radiusNormStep-1:
                        weightsIndex = self.radiusNormStep-1
                    toAdd = (weights[weightsIndex]**2)*[featureset]
                    inBandwidth += toAdd 

                newCentroid = np.average(inBandwidth,axis=0)
                newCentroids.append(tuple(newCentroid))

            uniques = sorted(list(set(newCentroids)))

            toPop = []

            for i in uniques:
                for ii in uniques:
                    if i == ii:
                        pass
                    elif np.linalg.norm(np.array(i)-np.array(ii))<= self.radius:
                        toPop.append(ii)
                        break

                for i in toPop:
                    try:
                        uniques.remove(i)
                    except:
                        pass

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

        self.classification = {}

        for i in range(len(self.centroids)):
            self.classification[i] = []

        for featureset in data:
            distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
            classification = distances.index(min(distances))
            self.classification[classification].append(featureset)

    def predict(self, data):
        distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification

clf = MeanShift()
clf.fit(X)

centroids = clf.centroids

for classification in clf.classification:
    color = colors[classification]
    for featureset in clf.classification[classification]:
        plt.scatter(featureset[0], featureset[1], marker="*", color=color, s=150, linewidths=5)

# for c in centroids:
#     plt.scatter(centroids[c][0], centroids[c][1], c="k", marker="*", s=150)

for c in centroids:
    plt.scatter(centroids[c][0], centroids[c][1], c="k", marker="*", s=150)

plt.show()