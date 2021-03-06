from sklearn import pipeline
from sklearn.preprocessing import (
    StandardScaler, PolynomialFeatures
)
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import GaussianNB


def gnb(max_poly_degree=3):
    """Return a naive bayes classifier"""
    steps = [
        ("polynomial_features", PolynomialFeatures(degree=max_poly_degree)),
        ("standard_scaler", StandardScaler()),
        ("gnb", GaussianNB())
    ]
    pipe = pipeline.Pipeline(steps=steps)

    estimator = GridSearchCV(
        pipe,
        {
            "polynomial_features__degree": range(1, max_poly_degree + 1)
        },
        scoring="log_loss"
    )

    return estimator
