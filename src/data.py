import pandas as pd
from urllib import urlretrieve
from os import path
from sklearn.cross_validation import train_test_split

DATA_SOURCE = ("https://drivendata.s3.amazonaws.com/data/2/public/"
                     "9db113a1-cdbe-4b1c-98c2-11590f124dd8.csv")
COMPETITION_DATA_SOURCE = ("https://drivendata.s3.amazonaws.com/data/2/"
                           "public/5c9fa979-5a84-45d6-93b9-543d1a0efc41.csv")

DATA_ROOT = path.join(path.dirname(__file__), "..", "data")
DATA_FILE = path.join(DATA_ROOT, "data.csv")
COMPETITION_DATA_FILE = path.join(DATA_ROOT, "competition.csv")


def download():
    """Download the data set"""
    urlretrieve(DATA_SOURCE, DATA_FILE)
    urlretrieve(COMPETITION_DATA_SOURCE, COMPETITION_DATA_FILE)


def normalize(df):
    """Normalize the dataframe"""
    return (df - df.mean()) / df.std()


def load(random_seed=1, test_ratio=0.25):
    """Load training and test sets. Feature dataframes are normalized, result
    vectors are not."""
    df = pd.read_csv(DATA_FILE, index_col=0)
    xtrain, xtest = train_test_split(
        df, random_state=random_seed, test_size=test_ratio)
    ytrain = xtrain.pop("Made Donation in March 2007")
    ytest = xtest.pop("Made Donation in March 2007")
    return normalize(xtrain), ytrain, normalize(xtest), ytest


def load_competition():
    df = pd.read_csv(COMPETITION_DATA_FILE, index_col=0)
    return normalize(df)
