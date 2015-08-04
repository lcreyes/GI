""" Main file to load config and run all classifiers and benchmarks
"""

import numpy as np
import pandas as pd


def load_data(fn):
    print(fn)
    df = pd.read_csv(fn)
    return df

# TODO stratifiying


def test_data():
    # from train_data get features and label, in the meantime random stuff
    NumberOfDataPoints = 100
    NumberOfTestPoints = 20
    NumberOfFeatures = 5

    data = np.random.rand(NumberOfDataPoints, NumberOfFeatures)
    test = np.random.rand(NumberOfTestPoints, NumberOfFeatures)
    labels = np.random.randint(2, size=NumberOfDataPoints)
    test_labels = np.random.randint(2, size=NumberOfTestPoints)

    return data, labels, test, test_labels