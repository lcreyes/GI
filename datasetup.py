""" Main file to load config and run all classifiers and benchmarks
"""

import pandas
import logging
import numpy as np
import sklearn.cross_validation
import sklearn.preprocessing

voya_logger = logging.getLogger('clairvoya')


def load_data(filename):
    """ Loads the data from the given filename in CSV format and scales the features between 0 and 1

    CSV file is expect to have the first row as headers, in the format

        id, label, X_1, X_2 ... X_n

    :param filename: of the CSV file

    :return: y, X_scaled
    """

    df = pandas.read_csv(filename)

    return df


def split_df_labels_features(df):
    """ splits a dataframe into y and X provided it conforms to the required input colums of id, labels, features...
    """

    y = df['label']
    X = df.iloc[:, 2:]  # first few columns are ids and the labels

    return y.values, X


def scale_features(X):
    """ Scales X to be between 0 and 1
    """
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    X_scaled = min_max_scaler.fit_transform(X)

    return X_scaled


def split_test_train_df_pu(df, test_size, pu_random_sampling_frac=False):
    """ Splits the data frame containing P, N and U labels into a training and testing dataframes, performing a
    random shuffle on the positives. If pu_random_sampling_frac is given then that % of unlabelled data is considered
    negatives in the training set and the rest is discarded

    testing =  testsize*P, N
    training = (1-testsize)*P, U

    In the process negative labels (-1) are converted to 0 in order to be compare the sets

    :param pu_random_sampling_frac: fraction of unlabelled (0 to 1) to be considered negatives (randomly sampled)
    """
    positives = df[df['label'] == 1]
    negatives = df[df['label'] == -1]
    unlabeled = df[df['label'] == 0]

    positives = positives.reindex(np.random.permutation(positives.index))

    num_positives_test = int(len(positives.index) * test_size)
    positives_test = positives[:num_positives_test]
    positives_train = positives[num_positives_test:]

    if pu_random_sampling_frac:
        unlabeled = unlabeled.sample(frac=pu_random_sampling_frac)

    df_train = positives_train.append(unlabeled, ignore_index=True)
    df_test = positives_test.append(negatives, ignore_index=True)

    assert set(df_train['label'].unique()) == set((1, 0))
    assert set(df_test['label'].unique()) == set((1, -1))

    # for comparisons unlabelled and negative must be the same value, the classifiers just treat them differently
    df_test.loc[df_test.label == -1, 'label'] = 0
    
    assert set(df_test['label'].unique()) == set((1, 0))

    return df_test, df_train


def get_stratifed_data(y, X, test_size):
    """ Splits the data into a training set and test set as defined in config
    :param y:
    :param X:
    :return:
    """
    sss = sklearn.cross_validation.StratifiedShuffleSplit(y, n_iter=1, test_size=test_size)

    train_index, test_index = next(iter(sss))

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    return X_train, y_train, X_test, y_test


def parse_config_module_name(config_name):
    """ Parses the config module name given as comanline input to the correct form for a python import. This involves
    changing '/' to '.' and removing '.py' from file names. Both regularly added by tab completion.

    :param config_name:
    :return:
    """

    return config_name.replace('/', '.').replace('.py', '')