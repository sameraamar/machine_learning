from sklearn import svm
import pandas as pd


def prepare_iris_DS():
    print("Loading iris DS...")
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    iris = pd.read_csv(url, names=["sepal length", "sepal width", "petal length", "petal width", "Species"])
    df = pd.DataFrame(iris, columns=["sepal length", "sepal width", "petal length", "petal width", "Species"])

    df.head()
    iris.head()

    print("Iris DS is Loaded")

    columns, labels = ["sepal length", "sepal width"], ["Iris-setosa", "Iris-virginica"]
    
    total = df.shape[0]
    df = df[df.Species.isin(labels)]
    X = df[columns]

    print("selected {0} entries out of {1} from the dataset based on labels {2}".format(len(X), total, str(labels)))

    Y = df[["Species"]]
    Y.loc[Y.Species != labels[0], 'Species'] = 0.0
    Y.loc[Y.Species == labels[0], 'Species'] = 1.0

    X = X.as_matrix()
    Y = Y.as_matrix().astype(float)

    return X, Y


X, Y = prepare_iris_DS()

rbf_svc = svm.SVC(kernel='rbf', gamma=0.1, C=0.1)
rbf_svc.fit(X, Y)





 