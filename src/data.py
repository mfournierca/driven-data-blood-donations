import pandas as pd
from urllib import urlretrieve
from os import path
from sklearn.cross_validation import train_test_split

DATA_SOURCE_URL = ("https://drivendata.s3.amazonaws.com/data/2/public/"
                   "9db113a1-cdbe-4b1c-98c2-11590f124dd8.csv")
DATA_ROOT = path.join(path.dirname(__file__), "..", "data")
DATA_FILE = path.join(DATA_ROOT, "data.csv")


def download():
    urlretrieve(DATA_SOURCE_URL, DATA_FILE)


def load(random_seed=1, test_ratio=0.25):
    df = pd.read_csv(DATA_FILE, index_col=0)
    xtrain, xtest = train_test_split(
        df, random_state=random_seed, test_size=test_ratio)
    ytrain = xtrain.pop("Made Donation in March 2007")
    ytest = xtest.pop("Made Donation in March 2007")
    return xtrain, ytrain, xtest, ytest
