""" Main file to load config and run all classifiers and benchmarks
"""

import pandas
import sklearn.cross_validation
import sklearn.preprocessing
import voya_config


def load_data(filename):
    """ Loads the data from the given filename in CSV format and scales the features between 0 and 1

    CSV file is expect to have the first row as headers, in the format

        id, label, X_1, X_2 ... X_n

    :param filename: of the CSV file

    :return: y, X_scaled
    """

    print('loading data from: {}'.format(filename))
    df = pandas.read_csv(filename)

    y = df['label']
    X = df.iloc[:, 2:]  # first few columns are ids and the labels

    # feature scaling TODO outside this function?
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    X_scaled = min_max_scaler.fit_transform(X)

    return y.values, X_scaled


def get_stratifed_data(y, X):
    """ Splits the data into a training set and test set as defined in config
    :param y:
    :param X:
    :return:
    """
    # TODO (ryan) is the config best defined here or past as arugments to the function?
    sss = sklearn.cross_validation.StratifiedShuffleSplit(y, n_iter=1, test_size=voya_config.config['test_size'])

    train_index, test_index = next(iter(sss))

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    return X_train, y_train, X_test, y_test