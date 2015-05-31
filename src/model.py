import numpy
from sklearn import pipeline, decomposition
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


def logloss(model, x, y):
    return log_loss(y, model.predict_proba(x))


def model(x, y):
    """Return a model fit to the provided data.

    :return: a scikit-learn model
    :rtype: sklearn.pipeline
    """
    steps = [
        ("pca", decomposition.PCA()),
        ("logistic", LogisticRegression(C=0.5))
    ]
    pipe = pipeline.Pipeline(steps=steps)

    regularization_param_space = numpy.logspace(-4, 1)
    pca_n_components_space = range(1, len(x.columns))

    estimator = GridSearchCV(
        pipe,
        {
            "pca__n_components": pca_n_components_space,
            "logistic__C": regularization_param_space
        }
    )
    estimator.fit(x, y)

    return estimator.best_estimator_
