""" PU Learning classifiers
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class PosOnly(BaseEstimator, TransformerMixin):
    """
    Adapts any probabilistic binary classifier to positive-unlabled learning using the PosOnly method proposed by
    Elkan and Noto:

    Elkan, Charles, and Keith Noto. \"Learning classifiers from only positive and unlabeled data.\"
    Proceeding of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2008.

    Adapted from code by Alexandre Drouin from https://github.com/aldro61/pu-learning which is
    Copyright 2013 Alexandre Drouin All rights reserved (see https://github.com/aldro61/pu-learning/blob/master/LICENSE.md)
    """

    def __init__(self, estimator, hold_out_ratio=0.1, precomputed_kernel=False):
        """
        estimator -- An estimator of p(s=1|x) that must implement:
                     * predict_proba(X): Takes X, which can be a list of feature vectors or a precomputed
                                         kernel matrix and outputs p(s=1|x) for each example in X
                     * fit(X,y): Takes X, which can be a list of feature vectors or a precomputed
                                 kernel matrix and takes y, which are the labels associated to the
                                 examples in X
        hold_out_ratio -- The ratio of training examples that must be held out of the training set of examples
                          to estimate p(s=1|y=1) after training the estimator
        precomputed_kernel -- Specifies if the X matrix for predict_proba and fit is a precomputed kernel matrix
        """
        self.estimator = estimator
        self.c = 1.0
        self.hold_out_ratio = hold_out_ratio

        if precomputed_kernel:
            self.fit = self.__fit_precomputed_kernel
        else:
            self.fit = self.__fit_no_precomputed_kernel

        self.estimator_fitted = False

    def __str__(self):
        class_string = 'PosOnly(p(s=1|y=1,x) ~= {}, Fitted: {}\n' \
                       '        Estimator: {}, \n)'.format(self.estimator, self.c, self.estimator_fitted)

        return class_string

    def __fit_precomputed_kernel(self, X, y):
        """
        Fits an estimator of p(s=1|x) and estimates the value of p(s=1|y=1) using a subset of the training examples

        X -- Precomputed kernel matrix
        y -- Labels associated to each example in X (Positive label: 1.0, Negative label: -1.0)
        """
        positives = np.where(y == 1.)[0]
        hold_out_size = np.ceil(len(positives) * self.hold_out_ratio)

        if len(positives) <= hold_out_size:
            raise ValueError('Not enough positive examples to estimate p(s=1|y=1,x). Need at least {}'.format(
                hold_out_size + 1))

        np.random.shuffle(positives)
        hold_out = positives[:hold_out_size]

        # Hold out test kernel matrix
        X_test_hold_out = X[hold_out]
        keep = list(set(np.arange(len(y))) - set(hold_out))
        X_test_hold_out = X_test_hold_out[:, keep]

        # New training kernel matrix
        X = X[:, keep]
        X = X[keep]

        y = np.delete(y, hold_out)

        self.estimator.fit(X, y)

        hold_out_predictions = self.estimator.predict_proba(X_test_hold_out)
        hold_out_predictions = hold_out_predictions[:, 1]  # Previously in a try / except

        c = np.mean(hold_out_predictions)
        self.c = c

        self.estimator_fitted = True

    def __fit_no_precomputed_kernel(self, X, y):
        """
        Fits an estimator of p(s=1|x) and estimates the value of p(s=1|y=1,x)

        X -- List of feature vectors
        y -- Labels associated to each feature vector in X (Positive label: 1.0, Negative label: -1.0)
        """
        positives = np.where(y == 1.)[0]
        hold_out_size = np.ceil(len(positives) * self.hold_out_ratio)

        if len(positives) <= hold_out_size:
            raise (
            'Not enough positive examples to estimate p(s=1|y=1,x). Need at least ' + str(hold_out_size + 1) + '.')

        np.random.shuffle(positives)
        hold_out = positives[:hold_out_size]
        X_hold_out = X[hold_out]
        X = np.delete(X, hold_out, 0)
        y = np.delete(y, hold_out)

        self.estimator.fit(X, y)

        hold_out_predictions = self.estimator.predict_proba(X_hold_out)

        hold_out_predictions = hold_out_predictions[:, 1]

        c = np.mean(hold_out_predictions)
        self.c = c

        self.estimator_fitted = True

        return self

    def predict_proba(self, X):
        """
        Predicts p(y=1|x) using the estimator and the value of p(s=1|y=1) estimated in fit(...)

        X -- List of feature vectors or a precomputed kernel matrix
        """

        if not self.estimator_fitted:
            raise Exception('The estimator must be fitted before calling predict_proba(...).')

        probabilistic_predictions = self.estimator.predict_proba(X)
        probabilistic_predictions[:, 1] /= self.c

        return probabilistic_predictions

    def predict(self, X, treshold=0.5):
        """
        Assign labels to feature vectors based on the estimator's predictions

        X -- List of feature vectors or a precomputed kernel matrix
        treshold -- The decision treshold between the positive and the negative class
        """
        if not self.estimator_fitted:
            raise Exception('The estimator must be fitted before calling predict(...).')

        return np.array([1. if p > treshold else -1. for p in self.predict_proba(X)])


class SVMDoubleWeight(object):
    """
    Runs the second approach described in Elkan & Noto (2008) for training on
    Positive +  Unlabeled (PU) data, namely:

    1. Run a classifier to obtain a probability estimate which is converted
       into a weight w(x) calculated following eqn 3 from section 3.
    2. For positive-labeled data use weight=1. For unlabeled data consider
       every element twice: once with weight w(x) and once with weight 1-w(x).
    3. Learn on training data again

    """

    def __init__(self, estimator):
        """
        estimator -- An estimator (SVC) of p(s=1|x):
        """
        self.estimator = estimator

        self.unlabeled_weights = []
        self.positives = []
        self.unlabeled = []

        self.c = 1.0  # c value from paper = p(s=1|y=1)

        self.estimator_fitted = False

        self.weights_available = False

    def __str__(self):
        class_string = 'PosOnly(p(s=1|y=1,x) ~= {}, Fitted: {}\n' \
                       '        Estimator: {}, \n)'.format(self.estimator, self.c, self.estimator_fitted)
        return class_string

    def _calculate_weights(self, X, y):

        # self.estimator = sklearn.svm.SVC(C=2.5, kernel='linear',
        #                               class_weight='auto', probability=True)


        self.estimator.fit(X, y)
        positive_probabilities = self.estimator.predict_proba(X[self.positives])[:, 1]
        unlabeled_probabilities = self.estimator.predict_proba(X[self.unlabeled])[:, 1]

        # c value from paper = p(s=1|y=1), three possible estimators: e1, e2, e3
        # TODO (ryan) allow the setting of c from init? for gridsearching?
        e1 = np.mean(positive_probabilities)
        e2 = np.sum(positive_probabilities) / (np.sum(positive_probabilities) + np.sum(unlabeled_probabilities))
        e3 = np.max(positive_probabilities)

        self.c = e1

        self.unlabeled_weights = (1 - self.c) / self.c * unlabeled_probabilities / (1.0 - unlabeled_probabilities)

        self.weights_available = True

    def fit(self, X, y):

        self.positives = np.where(y == 1)[0]
        self.unlabeled = np.where(y == 0)[0]

        num_positives = np.size(y[self.positives])
        num_unlabeled = np.size(y[self.unlabeled])

        if not self.weights_available:
            self._calculate_weights(X, y)

        # define new X set with unlabeled data added twice
        newX = np.vstack((X[self.positives], X[self.unlabeled], X[self.unlabeled]))

        # define new y set assigning "1" labels for positive data, and for unlabeled
        # dat: use "1"  for the first set, and then "0" labels for second set
        newy = np.concatenate((np.ones(num_positives),
                               np.ones(num_unlabeled), np.zeros(num_unlabeled)))

        weights = np.concatenate((np.ones(num_positives),
                                  self.unlabeled_weights, 1. - self.unlabeled_weights))

        self.estimator.C = 1.0
        self.estimator.class_weight = None

        self.estimator.fit(newX, newy, sample_weight=weights)

        self.estimator_fitted = True

        return self.estimator

    def predict_proba(self, X):
        """
        Predicts p(y=1|x) using the estimator and the value of p(s=1|y=1) estimated in fit(...)

        X -- List of feature vectors or a precomputed kernel matrix
        """

        if not self.estimator_fitted:
            raise Exception('The estimator must be fitted before calling predict_proba(...).')

        probabilistic_predictions = self.estimator.predict_proba(X)

        return probabilistic_predictions

    def predict(self, X, treshold=0.5):
        """
        Assign labels to feature vectors based on the estimator's predictions

        X -- List of feature vectors or a precomputed kernel matrix
        treshold -- The decision treshold between the positive and the negative class
        """
        if not self.estimator_fitted:
            raise Exception('The estimator must be fitted before calling predict(...).')

        return np.array([1. if p > treshold else -1. for p in self.predict_proba(X)])