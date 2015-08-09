import numpy as np
from sklearn import pipeline, decomposition
from sklearn.preprocessing import (
    StandardScaler, PolynomialFeatures
)
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression


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

