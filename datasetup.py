""" Main file to load config and run all classifiers and benchmarks
"""

import pandas
import sklearn.cross_validation
import sklearn.preprocessing
import voya_config


def load_data(filename):
    """ Loads the data from the given filename in CSV format and scales the features between 0 and 1

    CSV file is expect to have the first row as headers, in the format

        id, id, added__orig_index, cid, label, X_1, X_2 ... X_n

    :param filename: of the CSV file

    :return: y, X_scaled
    """

    print('loading data from: {}'.format(filename))
    df = pandas.read_csv(filename)

    y = df['label']
    X = df.iloc[:, 4:]  # first few columns are ids and the labels

    # feature scaling TODO outside this function?
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    X_scaled = min_max_scaler.fit_transform(X)

    return y.values, X_scaled


def split_train_data(y):  # TODO is there a good reason for a wrapper that just fills in the config?
    """ Splits the data into traing and test sets using the values given in the config file.

    :param y: labels
    :return: list of size <num_split_iterations> eahc with (train_index, test_index)
    """
    sss = sklearn.cross_validation.StratifiedShuffleSplit(y, n_iter=voya_config.config['num_split_iterations'],
                                                      test_size=voya_config.config['test_size'],
                                                      random_state=voya_config.config['random_seed'])
    return sss
