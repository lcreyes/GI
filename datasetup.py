""" Main file to load config and run all classifiers and benchmarks

Notes, currently its assumed data files loaded are structured, id, label, X_1, X_2 ... X_n this is hard coded in both
split_df_labels_features and scale_dataframe_features
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


def downsample_pu_df(df, unlab_to_pos_ratio):
    """ Takes input of the universe with P 0 N -1 and U 0 data and randomly samples the universe

    :param df: input data frame with positives, negatives and unlabelled data
    :param unlab_to_pos_ratio: how much unlabelled to include as a propotion of positives

    :return: df containing a random sample of unlablled at the ratio specified
    """
    num_labels = df.label.value_counts()  # numbers in each group
    num_pos = num_labels[1]
    num_unl = num_labels[0]

    num_required_unlab = int(num_pos * unlab_to_pos_ratio)
    assert(num_required_unlab <= num_unl), 'Operation would require {} unlabelled, we have {}'.format(num_required_unlab,
                                                                                                      num_unl)

    df_pos_neg = df[df['label'] != 0]
    df_unlab = df[df['label'] == 0]


    df_unlab.reindex(np.random.permutation(df_unlab.index))
    downsampled_df = df_pos_neg.append(df_unlab[:num_required_unlab], ignore_index=True)

    voya_logger.info('Downsampled from {} to {} unlabelled'.format(num_unl, num_required_unlab))

    return downsampled_df


def split_df_labels_features(df):
    """ splits a dataframe into y and X provided it conforms to the required input colums of id, labels, features...
    """

    y = df['label']
    X = df.iloc[:, 2:]  # first few columns are ids and the labels

    return y.values, X.values


def scale_dataframe_features(df):
    """ Performes a Min Max scaling between 0 and 1 to a whole dataframe
    :param df:
    :return: pandas.core.frame.DataFrame
    """

    df[df.columns[2:]] = df[df.columns[2:]].apply(lambda x: sklearn.preprocessing.MinMaxScaler().fit_transform(x))


def scale_features(X):
    """ Scales X to be between 0 and 1
    """
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    X_scaled = min_max_scaler.fit_transform(X)

    return X_scaled


def split_test_train_df_pu(df, test_size, test_neg_to_pos_ratio = None):
    """ Splits the data frame containing P, N and U labels into a training and testing dataframes, performing a
    random shuffle on the positives.

    testing =  testsize*P, N
    training = (1-testsize)*P, U

    In the process negative labels (-1) are converted to 0 in order to be compare the sets

    :param pu_random_sampling_frac: fraction of unlabelled (0 to 1) to be considered negatives (randomly sampled)
    """
    positives = df[df['label'] == 1]
    negatives = df[df['label'] == -1]
    unlabeled = df[df['label'] == 0]
    
    positives = positives.reindex(np.random.permutation(positives.index))
    negatives = negatives.reindex(np.random.permutation(negatives.index))
 
    num_positives_test = int(len(positives.index) * test_size)
    positives_test = positives[:num_positives_test]
    positives_train = positives[num_positives_test:]

    if test_neg_to_pos_ratio is not None:
        num_negatives_test = int(num_positives_test*test_neg_to_pos_ratio)
    else:
        num_negatives_test = len(negatives.index)
        
    negatives_test = negatives[:num_negatives_test]
        

    df_train = positives_train.append(unlabeled, ignore_index=True)
    df_test = positives_test.append(negatives_test, ignore_index=True)

    assert set(df_train['label'].unique()) == set((1, 0)), df_train.label.value_counts()
    assert set(df_test['label'].unique()) == set((1, -1)), df_test.label.value_counts()

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