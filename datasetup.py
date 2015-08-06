""" Main file to load config and run all classifiers and benchmarks
"""

import pandas
import sklearn.cross_validation
import sklearn.preprocessing
import voya_config


def load_data(filename):
    print(filename)
    df = pandas.read_csv(filename)
    label=df['label']
    features=df.iloc[:, 4:19]
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    features_scaled = min_max_scaler.fit_transform(features)
    #features_scaled = preprocessing.scale(features.values)
    return label.values, features_scaled


def split_train_data(y):  # TODO control with config, move to runner / add value with real df
    sss = sklearn.cross_validation.StratifiedShuffleSplit(y, n_iter=voya_config.config['number_of_folds'],
                                                      test_size=voya_config.config['fraction_of_test_events'],
                                                      random_state=voya_config.config['random_seed'])
    return sss
