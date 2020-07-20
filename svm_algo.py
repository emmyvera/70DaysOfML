import matplotlib.pyplot as plt 
from matplotlib import style
import numpy as np 
style.use("ggplot")

class SupportVectorMachine:
    def __init__(self, visualization =True):
        self.visualization = visualization
        self.colors = {1:"r", -1:"b"}

        # Visualization Setting
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)

    #Trainind Code Finding our w and b
    def fit(self, data):
        # get the data passed
        self.data = data
        # Collect  {|w| : [w,b]}
        optDict = {}

        #This Transform will be applied to the vector of w as we step each time
        transform = [[1,1],
                    [-1,1],
                    [-1,-1],
                    [1,-1]]
        # To store our features in other to get the maximun and minimum ranges for our graph
        allData = []
        
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    allData.append(feature)

        self.maxFeatureValue = max(allData)
        self.minFeatureValue = min(allData)
        allData = None


        #Taking steps to get to get to our global minium (more like learing rate in gradient decsent)
        step_sizes = [self.maxFeatureValue * 0.1,
                        self.maxFeatureValue * 0.01,
                        self.maxFeatureValue * 0.001,
                        #Please remove this steps if your computer becomes slow, just 0.1, 0.01, 0.001 will do
                        self.maxFeatureValue * 0.0002,
                        self.maxFeatureValue * 0.0001]

        bRangeMultiple = 5

        # We don't need to take small steps with b as we did with w
        bMultiply = 5

        # First element of the vector w
        latestOptimum = self.maxFeatureValue*10

        for step in step_sizes:
            w = np.array([latestOptimum,latestOptimum])

            optimized = False
            while not optimized:
                for b in np.arange(-1*(self.maxFeatureValue*bRangeMultiple), 
                                        self.maxFeatureValue*bRangeMultiple,
                                            step=bMultiply):
                    for transformation in transform:
                        w_t = w*transformation
                        foundOption = True
                        #weakest link in svm fundamentals 
                        #yi(xi.w+b) >= 1
                        for i in self.data:
                            for xi in self.data[i]:
                                yi=i
                                if not yi*(np.dot(w_t,xi)+b) >= 1:
                                    foundOption = False

                        if foundOption:
                            optDict[np.linalg.norm(w_t)] = [w_t, b]

                if w[0] < 0:
                    optimized = True
                    print("Optimized a Step")
                else:
                    w = w - step

            norms = sorted([n for n in optDict])
            optChoice = optDict[norms[0]]
            self.w = optChoice[0]
            self.b = optChoice[1]

            latestOptimum = optChoice[0][0]+step*2

        for i in self.data:
            for xi in self.data[i]:
                yi = i
                print(xi, " : ", yi*(np.dot(self.w, xi) + self.b))

    def predict(self, features):
        # (x.w+b) where x = features passed, w = weight, b = bias
        classification = np.sign(np.dot(np.array(features), np.array(self.w)) + self.b)
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=100, marker="*", c=self.colors[classification] )
        return classification

    def visualize(self):
        [[self.ax.scatter(x[0],x[1], s=100, color=self.colors[i]) for x in data_dict[i]] for i in data_dict]


        # hyperplane = x.w+b
        # v = x.w+b
        #positive surport vector = 1
        #negative surport vector = -1
        # decision boundary = 0
        def hyperplane(x,w,b,v):
            return (-w[0]*x-b+v) / w[1]

        dataRange = (self.minFeatureValue*0.9,self.maxFeatureValue*1.1)
        hypXMin = dataRange[0]
        hypXMax = dataRange[1]

        #Positive Support Vector
        # x.w + b = 1
        psv1 = hyperplane(hypXMin,self.w,self.b,1)
        psv2 = hyperplane(hypXMax,self.w,self.b,1)
        self.ax.plot([hypXMin, hypXMax], [psv1,psv2], "k")

        #Negative Support Vector
        # x.w + b = -1
        nsv1 = hyperplane(hypXMin,self.w,self.b,-1)
        nsv2 = hyperplane(hypXMax,self.w,self.b,-1)
        self.ax.plot([hypXMin, hypXMax], [nsv1,nsv2], "k")

        #Decision Boundary
        # x.w + b = 0
        db1 = hyperplane(hypXMin,self.w,self.b,0)
        db2 = hyperplane(hypXMax,self.w,self.b,0)
        self.ax.plot([hypXMin, hypXMax], [db1,db2], "w--")

        plt.show()

data_dict = {-1: np.array([ [1,7],
                            [2,8],
                            [3,8],]), 

            1: np.array([[5,1],
                         [6,-1],
                         [7,3],]) }

svm = SupportVectorMachine()
svm.fit(data=data_dict)

predictThisData = [[1,3],
                    [4,-3],
                    [7,2],
                    [13,-9],
                    [-11,17],
                    [6,2],
                    [-10,5]]

for p in predictThisData:
    svm.predict(p)

svm.visualize()