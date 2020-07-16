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
                    [-1,-1]
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
                        self.maxFeatureValue * 0.001]

        bRangeMultiple = 5

        # We don't need to take small steps with b as we did with w
        bMultiply = 5

        # First element of the vector w
        latestOptimum self.maxFeatureValue*10

        for step in step_sizes:
            w = np.array([latestOptimum,latestOptimum])

            optimized = False
            while not optimized:
                for b in np.arange(-1*(self.maxFeatureValue*bRangeMultiple), 
                                            maxFeatureValue*bRangeMultiple,
                                            step=bMultiply):
                    for transformation in transform:
                        w_t = w*transformation
                        foundOption = True
                        #weakest link in svm fundamentals 
                        #yi(xi.w+b) >= 1
                        for i in self.data:
                            for xi in self.data[i]:
                                yi=i
                                if not yi*(np.dot(w_t,xi)+b >= 1):
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
            self.w = optChoice[1]
            self.b = optChoice[2]

            latestOptimum = optChoice[0][0]+step*2

    def predict(self, features):
        # (x.w+b) where x = features passed, w = weight, b = bias
        classification = np.sign(np.dot(np.array(features), np.array(self.w)) + self.b)
        return classification

data_dict = {-1: np.array([ [1,7],
                            [2,8],
                            [3,8],]), 

            1: np.array([[5,1],
                         [6,-1],
                         [7,3],]) }

