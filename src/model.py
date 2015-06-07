import numpy
from pandas import DataFrame
from sklearn import pipeline, decomposition
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import BernoulliRBM
from sklearn.cross_validation import train_test_split


def logloss(model, x, y):
    return log_loss(y, model.predict_proba(x))


def get_learning_curve(model, x, y, niter=None, stepsize=100, trainsize=0.75):
    """Train and evaluate a model on multiple training set sizes. Return a 
    data frame that can be used to create learning curves. 
    
    :param model: a method that takes a training set of features x and results
        y and returns an sklearn model with the fit() method
    :type model: function
    :param x: the full set of pre-processed and normalized feature vectors
    :type x: pandas.DataFrame
    :param y: the full set of result vectors
    :type y: pandas.DataFrame
    :param niter: the number of iterations to perform
    :type niter: int
    :param stepsize: the number of training example to increase by on each 
        iteration
    :type stepsize: int
    :param trainsize: the proportion of data points to use for the training set
    :type trainsize: float
    :returns: a dataframe
    """
   
    if niter is None:
        niter = int((len(x) * trainsize) / stepsize)

    columns = ["iteration", "n_train", "train_log_loss", "test_log_loss"]
    results = []
    n_smoothing_iterations = 5
 
    n_train = 0
    for i in range(niter):
        n_train += stepsize
        test_log_loss = 0
        train_log_loss = 0

        for j in range(n_smoothing_iterations):
            xtrain, xtest = train_test_split(x)
            xtrain = xtrain[:n_train]
            ytrain, ytest = y[xtrain.index], y[xtest.index]
            m = model(xtrain, ytrain)
            
            train_log_loss += logloss(m, xtrain, ytrain)
            test_log_loss += logloss(m, xtest, ytest) 

        results.append(
            dict(
                zip(
                    columns, 
                    [
                        i, 
                        n_train,
                        train_log_loss / n_smoothing_iterations,
                        test_log_loss / n_smoothing_iterations
                    ]
                )
            )
        )
    return DataFrame(results)


def gnb(x, y):
    """Return a naive bayes classifier fit to the provided data"""
    gnb = GaussianNB()
    gnb.fit(x, y)
    return gnb
    

def logistic(x, y):
    """Return a logistic model fit to the provided data."""
    steps = [
        ("pca", decomposition.PCA()),
        ("logistic", LogisticRegression(C=0.5))
    ]
    pipe = pipeline.Pipeline(steps=steps)

    estimator = GridSearchCV(
        pipe,
        {
            "pca__n_components": range(1, len(x.columns)),
            "logistic__C": numpy.logspace(-4, 1)
        },
        scoring="log_loss"
    )
    estimator.fit(x, y)

    return estimator.best_estimator_
