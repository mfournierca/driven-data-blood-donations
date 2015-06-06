import numpy
from sklearn import pipeline, decomposition
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import BernoulliRBM


def logloss(model, x, y):
    return log_loss(y, model.predict_proba(x))


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
