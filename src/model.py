import numpy as np

from pandas import DataFrame

from sklearn import pipeline, decomposition

from sklearn.preprocessing import (
    StandardScaler, PolynomialFeatures
)

from sklearn.grid_search import GridSearchCV
from sklearn.metrics import log_loss

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.learning_curve import learning_curve


# 
# util
# 

def get_learning_curve(
        model, 
        x, 
        y, 
        cv=3, 
        train_sizes=None, 
        scoring="log_loss"):
    """Get a dataframe representing the learning curve for a model
 
    :param model: a sklearn model
    :type model: object
    :param x: the full dataframe of features to pass to the model pipeline
    :type x: pandas.DataFrame
    :param y: the full vector of results
    :type y: pandas.DataFrame
    :param cv: the number of cross validation folds to make on each iteration
    :param train_sizes: a list of training set sizes to go through
    :returns: a dataframe
    """
 
    if train_sizes is None:
        train_sizes = range(50, 400, 25)
 
    sizes, train_score, cv_score = learning_curve(
        model, x, y, train_sizes=train_sizes, cv=cv, scoring=scoring
    )
    train_score = np.apply_along_axis(np.mean, 1, train_score)
    cv_score = np.apply_along_axis(np.mean, 1, cv_score)
    df = DataFrame(
        [sizes, train_score, cv_score], 
        index=["sizes", "train_score", "cv_score"]
    ).transpose()
    return df


#
# models 
#

def gnb(max_poly_degree=1):
    """Return a naive bayes classifier"""
    steps = [
        ("polynomial_features", PolynomialFeatures(degree=max_poly_degree)),
        ("standard_scaler", StandardScaler()),
        ("gnb", GaussianNB())
    ]
    pipe = pipeline.Pipeline(steps=steps)
    return pipe


def logistic(max_pca_components=4, max_poly_degree=1):
    """Return a logistic model"""
    steps = [
        ("polynomial_features", PolynomialFeatures()),
        ("standard_scaler", StandardScaler()),
        ("pca", decomposition.PCA()),
        ("logistic", LogisticRegression())
    ]
    pipe = pipeline.Pipeline(steps=steps)

    estimator = GridSearchCV(
        pipe,
        {
            "polynomial_features__degree": range(1, max_poly_degree + 1),
            "pca__n_components": range(1, max_pca_components),
            "logistic__C": np.logspace(-4, 1)
        },
        scoring="log_loss"
    )

    return estimator


def knn(max_k=33, max_poly_degree=1):
    """Return a k means clustering classifier"""

    steps = [
        ("polynomial_features", PolynomialFeatures()),
        ("knn", KNeighborsClassifier())
    ]
    pipe = pipeline.Pipeline(steps=steps)

    estimator = GridSearchCV(
        pipe,
        {
            "polynomial_features__degree": range(1, max_poly_degree + 1),
            "knn__n_neighbors": range(1, max_k)
        },
        scoring="log_loss"
    )

    return estimator

