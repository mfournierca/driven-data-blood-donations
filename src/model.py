import numpy
from sklearn import pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


def logloss(model, x, y):
    return log_loss(y, model.predict_proba(x))


def model():
    """Return a model. 

    :return: a scikit-learn model
    :rtype: sklearn.pipeline
    """
    steps = [("logistic", LogisticRegression())] 
    pipe = pipeline.Pipeline(steps=steps)

    regularization_param_space = numpy.logspace(-4, 0)
    estimator = GridSearchCV(
        pipe,
        {"logistic__C": regularization_param_space}
    )
    return estimator
