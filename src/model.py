from sklearn import pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


def model():
    """Return a scikit learn pipeline. 

    :return: a scikit-learn pipeline
    :rtype: sklearn.pipeline
    """
    steps = [("logistic", LogisticRegression())]
    return pipeline.Pipeline(steps=steps)
