""" Main file to load config and run all classifiers and benchmarks
"""

import numpy as np
import pandas as pd
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn import preprocessing
from config import config

def load_data(fn):
    print(fn)
    df = pd.read_csv(fn)
    label=df['label']
    features=df.iloc[:,4:19]
    features_scaled = preprocessing.scale(features.values)
    return label.values, features_scaled


def split_train_data(y):  # TODO control with config, move to runner / add value with real df
  sss = StratifiedShuffleSplit(y, n_iter=config['number_of_folds'], 
        test_size=config['fraction_of_test_events'], random_state=config['random_seed'])
  return sss



