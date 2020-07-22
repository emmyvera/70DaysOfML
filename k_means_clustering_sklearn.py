import matplotlib.pyplot as plt 
import numpy as np 
from matplotlib import style
style.use("ggplot")

from sklearn.cluster import KMeans

X = np.array([[1,2],
            [1.5,3],
            [6,8],
            [8,8],
            [1,0],
            [9,11]])

# plt.scatter(X[:,0], X[:,1], s=100)
# plt.show()

clf = KMeans(n_clusters=2)

clf.fit(X)

centroids = clf.cluster_centers_
label = clf.labels_

colors = ["g.","r.","c.","b.","k.","o."]

for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[label[i]], markersize=25)

plt.scatter(centroids[:,0], centroids[:,1], marker="x", s=100)
plt.show()