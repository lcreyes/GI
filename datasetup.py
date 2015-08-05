""" Main file to load config and run all classifiers and benchmarks
"""

import pandas as pd
import sklearn.cross_validation
import sklearn.preprocessing
import config
from sklearn import preprocessing


def load_data(fn):
    print(fn)
    df = pd.read_csv(fn)
    label=df['label']
    features=df.iloc[:,4:19]
    min_max_scaler = preprocessing.MinMaxScaler()
    features_scaled = min_max_scaler.fit_transform(features)
    #features_scaled = preprocessing.scale(features.values)
    return label.values, features_scaled



def split_train_data(y):  # TODO control with config, move to runner / add value with real df
    sss = sklearn.cross_validation.StratifiedShuffleSplit(y, n_iter=config.config['number_of_folds'],
                                                      test_size=config.config['fraction_of_test_events'],
                                                      random_state=config.config['random_seed'])
    return sss
