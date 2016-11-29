# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 18:22:07 2016

@author: SAMER AAMAR
"""
import numpy as np
import math
import matplotlib.pyplot as plt


#import pandas.tools.plotting as pplt
from pandas.tools.plotting import scatter_matrix, andrews_curves, parallel_coordinates
import pandas as pd

#Parameters

folder = "c:/temp/iris"
logger_level = 1
plot_vis = True

# helper functions

def info(message):
    if logger_level <= 1:
        print(message)
    

def debug(message):
    if logger_level <= 0:
        print(message)

def dotproduct(v1, v2):
    d = v1.dot(v2.T)
    return d[0,0] # get scalar
    #return sum((a*b) for a, b in zip(v1, v2))

def length(v):
    return math.sqrt(dotproduct(v, v))

# return angle in PI
def angle(v1, v2):
    d = (length(v1) * length(v2))
    if d == 0:
        return None
    return math.acos(dotproduct(v1, v2) / d) / math.pi

  
def generate_matrix(df, columns, labels):
    """
    generate a np.matrix objects from the given DataFrame object.
    :param df: DataFrame object
    :param columns: names of relevant columns to be considered
    :param labels: values of labels to filter iris dataset
    :return: (X, Y)
    X - data samples
    Y - data labels
    """
    total = df.shape[0]
    df = df[df.Species.isin(labels)]
    X = df[columns]

    info("selected {0} entries out of {1} from the dataset based on labels {2}".format(len(X), total, str(labels)))

    Y = df[["Species"]]
    Y.loc[Y.Species != labels[0], 'Species'] = 0.0
    Y.loc[Y.Species == labels[0], 'Species'] = 1.0

    X = X.as_matrix()
    Y = Y.as_matrix()

    return X, Y

def my_plot(X, Y, theta, k=None, i=None):
    """
    Plot X and Y and theta.

    if m is number of samples and n is number of features
    :param X: samples. dimension is (m,n)
    :param Y: labels. dimension is (m,1)
    :param theta: theta values. dimension is (1,n)
    :param i: index of a sample (optional)
    :return: None
    """
    global plot_vis, logger
    
    if X.shape[1] > 3:
        debug("Iteration {0}: sample {1}".format(k, i))
        return
    
    info('---- Interation {0} ----'.format(k) )
    debug("  X[{0}]: {1}".format( i, str(X[i, :])) )
    debug("  y[{0}]= {1}".format(i, str(Y[i][0]) ) )
        
    a = angle(np.asmatrix([1,0]) , theta[:, :-1])
    if a == None:
        a = float('inf')
    info("  theta: {0} angle to vector (1,0) is {1:.4f}*PI".format( theta, a ) )

    Ynew = f(X.dot(theta.T))
    if (i != None):
        debug("  Ynew[{0}] - 5 entries: {1}".format(i, str(Ynew[0:5, :])) )


    if not plot_vis:
        return
    
    plt.figure()
    #plt.hold(True)

    plotX1 = []
    plotY1 = []
    plotX2 = []
    plotY2 = []

    for s in range(X.shape[0]):
        if Ynew[s] == 0:
            plotX1.append(X[:, 0][s])
            plotY1.append(X[:, 1][s])
        else:
            plotX2.append(X[:, 0][s])
            plotY2.append(X[:, 1][s])

    plt.scatter(plotX1, plotY1, marker='o', color='r', cmap='prism')
    plt.scatter(plotX2, plotY2, marker='x', color='b', cmap='prism')
    #plt.scatter(X[:, 0], X[:, 1], marker="o", cmap='prism')
    #plt.hold(True)

    # draw a spearation line
    #tmp = theta / theta[0, theta.shape[1]-1]
    #p0 = theta.T * [0,0,1]
    #p1 = theta.T * [1,1,1]
    #plt.plot(p0[:, :-1], p1[:, :-1],  marker="o")

    plt.title('Neural Network - Iteration {0}\n<<Sample {1}>>'.format(k, i))
    plt.xlabel('X1')
    plt.ylabel('X2')

    # calculate the parameters of the separating line
    w0 = theta[0, 0]
    w1 = theta[0, 1]
    w2 = theta[0, 2]
    w0 /= -w1
    w2 /= -w1

    # determine the X axis range
    minx = float('inf')
    maxx = float('-inf')
    if len(plotX1)>0:
        minx = np.min(plotX1)
    if len(plotX2)>0:
        minx = min ( minx, np.min(plotX2))

    if len(plotX1)>0:
        maxx = np.max(plotX1)
    if len(plotX2)>0:
        maxx = max ( maxx, np.max(plotX2))

    xx = minx + (maxx - minx) * np.arange(10) / 10


    yy = xx * w0 + w2

    plt.plot(xx, yy, "-")
    print(xx, yy)
#      x2 = ( w1*x1+w3*x3 ) / -w2= x2


    #print(p0, p1)

    colors = "bry"
    #X[:, [0,1]]
    #plt.plot(p1[:-1], p1[:-1], ls="--", color="r")

    if k == None or i == None:
        plt.savefig(folder + "/plt.jpg")
    else:
        plt.savefig(folder + "/plt_{0:04d}_{1:04d}.jpg".format(k, i+1))
    #plt.show()

    plt.close()
  
# perceptron mapping function

def f(C):
    C[C>0] = 1
    C[C<0] = 0

    return C


def my_perceptron_train(X, Y):
    """
    The function should train the perceptron on the instances in matrix X.


    if m is number of samples and n is number of features
    :param X: X is matrix (instances x features). dimension of (m,n)
    :param Y: Y is a vector (of target values). dimension of (m,1)
    :return: 'theta' – the vector with the final parameters. dimension of (1,n)
             'k' – the number of updates.
    """

    k = 0
    eighta = 1
    
    X = np.c_[ X, np.ones( X.shape[0] ) ] 
    #theta = np.zeros( (1,X.shape[1]) )
    theta = np.random.random (X.shape[1] )
    theta = np.asmatrix(theta)

    #my_plot(X, Y, theta)

    changed = True
    while changed and k<500:
        changed = False
        for i in range(X.shape[0]):
            Xi = X[i, :]

            Yi = f(Xi.dot(theta.T))
            tmp =  (Y[i] - Yi)[0] # get a scalar
            #tmp = eighta * (tmp * Xi)
            theta_new = theta + ( eighta * (tmp * Xi) )
            
            if np.sum(theta - theta_new) != 0:
                changed = True

            theta = theta_new
        my_plot(X, Y, theta, k, i)

        k += 1

    #my_plot(X, Y, theta, k, i)


    return k, theta[: , 0:-1]



def my_perceptron_test(theta, X_test, y_test):
    """
    assume that n is number of features and m is number of samples
    :param theta: is the classification vector from my_perceptron_train(). dimension is (1,n)
    :param X_test: samples. dimension of (m,n)
    :param y_test: labels. dimension is (m,1)
    :return: y_res: which is classification answer. . dimension is (m,1)
             accuracy: float number representing the accuracy
    """
    Y_res = X_test.dot(theta.T)
    Y_res = f(Y_res)

    count = 0
    for i in range(Y_res.shape[0]):
        if Y_res[i,0] == y_test[i]:
            count += 1

    accuracy = count / Y_res.shape[0]

    return Y_res, accuracy




def test_or_NN():
    X = np.matrix('0 0; 0 1; 1 0; 1 1')
    Y = np.matrix('0, 1, 1, 1').T

    my_neural_network(X, Y, X, Y)

def test_xor_NN():
    X = np.matrix('0 0; 0 1; 1 0; 1 1')
    Y = np.matrix('0, 1, 1, 0').T

    my_neural_network(X, Y, X, Y)

def init_irisDS():
    info("Loading iris DS...")
    iris = pd.read_csv('iris_dataset/iris.data',
                       names=["sepal length", "sepal width", "petal length", "petal width", "Species"])
    df = pd.DataFrame(iris, columns=["sepal length", "sepal width", "petal length", "petal width", "Species"])

    df.head()
    iris.head()

    global plot_vis
    if plot_vis:
        plt.figure()
    
        andrews_curves(iris, 'Species')
        scatter_matrix(iris, alpha=0.2, figsize=(6, 6), diagonal='kde', marker="x")
        plt.savefig(folder + r"/plt_iris_ds.jpg")


    info("Iris DS is Loaded")
    return iris, df
    
def test_irisNN(train_df, test_df, columns, labels):
    xtrain, ytrain = generate_matrix(train_df, columns, labels)
    xtest, ytest = generate_matrix(test_df, columns, labels)
    my_neural_network(xtrain, ytrain, xtest, ytest)
    
def my_neural_network(xtrain, ytrain, xtest, ytest):
    """
    run neural network based on given xtrain / ytrain and plot checks aganist testx/testy.
    :param xtrain:
    :param ytrain:
    :param xtest:
    :param ytest:
    :return:
    """
    k, theta = my_perceptron_train(xtrain, ytrain)
    info('Done after {0} iterations'.format( k))
    info('Theta is: {0}'.format( str(theta) ))

    if len(xtest) > 0 :
        y_res, accuracy = my_perceptron_test(theta, xtest, ytest)
    
        debug("Y resulting: {0}".format( str(y_res)))
        info("Accuracy: {0:.4f}".format(accuracy))



if __name__ == '__main__':
    #test_xor_NN()
    test_or_NN()
    input("Press <Enter> key to continue...")
    iris, df = init_irisDS()
    input("Press <Enter> key to continue...")

    #from sklearn.model_selection import train_test_split
    #train, test = train_test_split(df, test_size = 0.2)
    ratio = 0.6
    msk = np.random.rand(len(df)) < ratio
    train = df[msk]
    test = df[~msk]

    test_irisNN(train, test, ["sepal length", "sepal width"], ["Iris-setosa", "Iris-virginica"])
    #input("Press <Enter> key to continue...")
    #test_irisNN(train, test, ["sepal length", "sepal width", "petal length", "petal width"], ["Iris-setosa", "Iris-virginica"])

    
