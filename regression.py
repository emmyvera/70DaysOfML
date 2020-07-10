from statistics import mean
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import style
import random

#style.use("fivethirtyeight")
#Creating Point for testing
def createTest(hm, variance, step=2, correlation=False):
    val=1
    ys=[]
    for i in range(hm):
        y= val+random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == "pos":
            val+=step
        elif correlation and correlation == "neg":
            val-=step
        xs = [i for i in range(len(ys))]

    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


#xs = np.array([1,2,3,4,5,6])
#ys = np.array([5,6,4.5,6,7,8])

def bestFitGradient(xs, ys):
    m = ( ((mean(xs)* mean(ys)) - mean(xs*ys)) / 
    ( (mean(xs) * mean(xs)) - mean(xs*xs) ) )

    return m 

def bestIntercept(xs, ys):
    b = ( mean(ys) - (bestFitGradient(xs, ys) * mean(xs)))
    return b

def squaredError(ysLine, ysOrig):
    return sum((ysLine - ysOrig)**2)

def coefficientOfDetermination(ysLine, ysOrig):
    yMeanLine = [mean(ysOrig) for y in ysOrig]
    squareErrorReg = squaredError(ysLine, ysOrig)
    squaredErrorYMean = squaredError(yMeanLine, ysOrig)
    return 1 - (squareErrorReg/squaredErrorYMean)

xs, ys = createTest(40,10, correlation="neg")

m = bestFitGradient(xs, ys)
b = bestIntercept(xs,ys)


regressionLine = [(m*x) + b for x in xs]

predict_x = 9
predict_y = (m*predict_x + b)

plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, c="g")
plt.plot(xs, regressionLine)
plt.show()

print(bestFitGradient(xs, ys))
print(bestIntercept(xs, ys))
print(predict_y)
print(squaredError(regressionLine, ys))
print(coefficientOfDetermination(regressionLine,ys))