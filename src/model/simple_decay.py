import numpy as np
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn import pipeline
from sklearn.grid_search import GridSearchCV


class SimpleDecay(object):

    decay_factor = -1.0
    threshold = 0.5

    def __init__(self, decay_factor=None):
        if decay_factor:
            self.decay_factor = decay_factor

    def fit(self, x, y):
        pass

    def predict_proba(self, x):
        p = x["Number of Donations"] / x["Months since First Donation"]
        p = np.exp(self.decay_factor*(1.0/p)*x["Months since Last Donation"])
        return np.array(1 - p, p).transpose()

    def predict(self, x):
        p = self.predict_proba(x)
        return np.ndarray([1 if j > self.threshold else 0 for j in p])

    def score(self, x, y, scoring="log_loss"):
        if scoring != "log_loss":
            raise ValueError()
        return log_loss(y, self.predict_proba(x))


def simple_decay():
    """Return a simple decay model"""
    steps = [
        ("simple_decay", SimpleDecay())
    ]
    pipe = pipeline.Pipeline(steps=steps)

    estimator = GridSearchCV(
        pipe,
        {
            "simple_decay__decay_factor": np.arange(-10, -0.01, 0.25)
        },
        scoring="log_loss"
    )
    return estimator
