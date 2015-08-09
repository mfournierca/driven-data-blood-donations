from sklearn import pipeline
from sklearn.preprocessing import (
    StandardScaler, PolynomialFeatures
)
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


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

