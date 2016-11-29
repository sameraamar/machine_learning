# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 18:22:07 2016

@author: SAMER AAMAR
"""
import numpy as np
import math
import matplotlib.pyplot as plt


from pandas.tools.plotting import scatter_matrix #, parallel_coordinates
import pandas as pd

#Parameters

graphs_folder = "c:/temp/iris" # folder for the graphs
iris_ds_file = 'iris_dataset/iris.data'
logger_level = 1     # 1 = info, 0 = debug
plot_vis = True      # plot graphs or not
max_iterations = 500 # maximum allowed iterations

#-----------------------------------------
# helper functions
def info(message):
    if logger_level <= 1:
        print message 
    

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

def my_plot(X, Y, theta, k=None):
    """
    Plot X and Y and theta.

    if m is number of samples and n is number of features
    :param X: samples. dimension is (m,n)
    :param Y: labels. dimension is (m,1)
    :param theta: theta values. dimension is (1,n)
    :param i: index of a sample (optional)
    :return: None
    """
    global plot_vis
    
    info('---- Interation {0} ----'.format(k) )
        
    a = angle(np.asmatrix([1,0]) , theta[:, :-1])
    if a == None:
        a = float('inf')
    info("  theta: {0} angle to vector (1,0) is {1:.4f}*PI".format( theta, a ) )

    Ynew = f(X.dot(theta.T))

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

    plt.title('Neural Network - Iteration {0}'.format(k))
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

    xx = minx + (maxx - minx) * np.arange(10) / 10.0


    yy = xx * w0 + w2

    plt.plot(xx, yy, "-")
    if k == None:
        plt.savefig(graphs_folder + "/plt.jpg")
    else:
        plt.savefig(graphs_folder + "/plt_{0:04d}.jpg".format(k))

    plt.close()
  
#-----------------------------------------


# perceptron mapping function
def f(C):
    C[C>0] = 1.0
    C[C<0] = 0.0

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
    theta = np.zeros( X.shape[1] )
    #theta = np.random.random ( X.shape[1] )
    theta = np.asmatrix(theta)

    changed = True
    while changed and k<500:
        changed = False
        for i in range(X.shape[0]):
            Xi = X[i, :]

            Yi = f(Xi.dot(theta.T))
            tmp =  (Y[i] - Yi)[0] # get a scalar
            theta_new = theta + ( eighta * (tmp * Xi) )
            
            if np.sum(theta - theta_new) != 0:
                changed = True

            theta = theta_new
        my_plot(X, Y, theta, k)

        k += 1


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

    accuracy = 1.0 * count / Y_res.shape[0]

    return Y_res, accuracy




def buildNN_ORFunc():
    X = np.matrix('0 0; 0 1; 1 0; 1 1')
    Y = np.matrix('0, 1, 1, 1').T

    build_neural_network(X, Y, X, Y)

def buildNN_XORFunc():
    X = np.matrix('0 0; 0 1; 1 0; 1 1')
    Y = np.matrix('0, 1, 1, 0').T

    build_neural_network(X, Y, X, Y)

def load_irisDS():
    info("Loading iris DS...")
    iris = pd.read_csv(iris_ds_file, names=["sepal length", "sepal width", "petal length", "petal width", "Species"])
    df = pd.DataFrame(iris, columns=["sepal length", "sepal width", "petal length", "petal width", "Species"])

    df.head()
    iris.head()

    global plot_vis
    if plot_vis:
        plt.figure()
    
        #andrews_curves(iris, 'Species')
        scatter_matrix(iris, alpha=0.2, figsize=(6, 6), diagonal='kde', marker="x")
        plt.savefig(graphs_folder + r"/plt_iris_ds.jpg")


    info("Iris DS is Loaded")
    return iris, df
    
def buildNN_irisDS(train_df, test_df, columns, labels):
    xtrain, ytrain = generate_matrix(train_df, columns, labels)
    xtest, ytest = generate_matrix(test_df, columns, labels)
    build_neural_network(xtrain, ytrain, xtest, ytest)
    
def build_neural_network(xtrain, ytrain, xtest, ytest):
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
        info("Accuracy is: {0:.4f}".format(accuracy))



if __name__ == '__main__':
    #buildNN_XORFunc()
    buildNN_ORFunc()
    raw_input ("Press <Enter> key to continue...")
    iris, df = load_irisDS()
    raw_input("Press <Enter> key to continue...")

    ratio = 0.8
    info("Ratio selected is: {0}".format(ratio))
    
    msk = np.random.rand(len(df)) < ratio
    train = df[msk]
    test = df[~msk]

    buildNN_irisDS(train, test, ["sepal length", "sepal width"], ["Iris-setosa", "Iris-virginica"])

    
