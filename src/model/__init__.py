import numpy as np
from pandas import DataFrame
from sklearn.learning_curve import learning_curve

from knn import knn
from logistic import logistic
from gnb import gnb


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

