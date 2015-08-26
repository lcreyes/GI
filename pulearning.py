""" PU Learning classifiers
"""

import logging
import numbers
import itertools

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection.from_model import _LearntSelectorMixin
from sklearn.ensemble.base import _partition_estimators
import sklearn.cross_validation
import sklearn.ensemble
import sklearn.svm

from sklearn.utils import check_random_state, check_X_y
from sklearn.utils.random import sample_without_replacement
from sklearn.utils.fixes import bincount
from sklearn.externals.joblib import Parallel, delayed

MAX_INT = np.iinfo(np.int32).max

voya_logger = logging.getLogger('clairvoya')

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
            raise ValueError('Not enough positive examples to estimate p(s=1|y=1,x). Need at least ' + str(hold_out_size + 1) + '.')

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

    def predict(self, X):
        """
        Assign labels to feature vectors based on the estimator's predictions

        X -- List of feature vectors or a precomputed kernel matrix
        treshold -- The decision treshold between the positive and the negative class
        """
        if not self.estimator_fitted:
            raise Exception('The estimator must be fitted before calling predict(...).')


        return self.estimator.predict(X)


class PULearnByDoubleWeighting(BaseEstimator, TransformerMixin):
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
        estimator -- An estimator of p(s=1|x):
        """
        self.estimator = estimator

        self.unlabeled_weights = []
        self.positives = []
        self.unlabeled = []

        self.c = 1.0  # initialisation value from paper = p(s=1|y=1)

        self.estimator_fitted = False

      
    def __str__(self):
        class_string = 'DoubleWeight (p(s=1|y=1,x) ~= {}, Fitted: {}\n' \
                       '        Estimator: {}, \n)'.format(self.estimator, self.c, self.estimator_fitted)
        return class_string

    def _calculate_weights(self, X, y):

        # self.estimator = sklearn.svm.SVC(C=2.5, kernel='linear',
        #                               class_weight='auto', probability=True)


        self.estimator.fit(X, y)
        positive_probabilities = self.estimator.predict_proba(X[self.positives])[:, 1]
        unlabeled_probabilities = self.estimator.predict_proba(X[self.unlabeled])[:, 1]

        # c value from paper = p(s=1|y=1), three possible estimators: e1, e2, e3
        e1 = np.mean(positive_probabilities)
        # e2 = np.sum(positive_probabilities) / (np.sum(positive_probabilities) + np.sum(unlabeled_probabilities))
        # e3 = np.max(positive_probabilities)

        self.c = e1

        return (1 - self.c) / self.c * unlabeled_probabilities / (1.0 - unlabeled_probabilities)
        
    def fit(self, X, y):

        self.positives = np.where(y == 1)[0]
        self.unlabeled = np.where(y == 0)[0]

        num_positives = np.size(y[self.positives])
        num_unlabeled = np.size(y[self.unlabeled])

        unlabeled_weights = self._calculate_weights(X, y)

        # define new X set with unlabeled data added twice
        newX = np.vstack((X[self.positives], X[self.unlabeled], X[self.unlabeled]))

        # define new y set assigning "1" labels for positive data, and for unlabeled
        # dat: use "1"  for the first set, and then "0" labels for second set
        newy = np.concatenate((np.ones(num_positives),
                               np.ones(num_unlabeled), np.zeros(num_unlabeled)))

        weights = np.concatenate((np.ones(num_positives),
                                  unlabeled_weights, 1. - unlabeled_weights))

        if hasattr(self.estimator, 'C'):
            self.estimator.C = 1.0
        
        if hasattr(self.estimator, 'class_weight'):
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

    def predict(self, X):
        """
        Assign labels to feature vectors based on the estimator's predictions

        X -- List of feature vectors or a precomputed kernel matrix
        treshold -- The decision treshold between the positive and the negative class
        """
        if not self.estimator_fitted:
            raise Exception('The estimator must be fitted before calling predict(...).')

        return self.estimator.predict(X)


class PUBagging(sklearn.ensemble.BaggingClassifier,BaseEstimator, TransformerMixin):
    """
    Runs the bagging approach suggested by Mordelet & Vert (2010), namely:

    INPUT: Positive set, unlabeled set, fraction of unlabeled set in bag (K), number of bootstraps (T)
    for t=1 to T do:
        Draw a subsample of U with size K*sizeof(U)
        Train a classifier C_i to discriminate P against U
    Return 1/T * Sum (C_i)

    """
                
    def fit(self, X, y, sample_weight=None):
        """Build a Bagging ensemble of estimators from the training
           set (X, y).
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.
        y : array-like, shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).
        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if the base estimator supports
            sample weighting.
        Returns
        -------
        self : object
            Returns self.
        """
        random_state = check_random_state(self.random_state)

        # Convert data
        X, y = check_X_y(X, y, ['csr', 'csc', 'coo'])

        # Remap output
        n_samples, self.n_features_ = X.shape
        y = self._validate_y(y)

        # Check parameters
        self._validate_estimator()

        if isinstance(self.max_samples, (numbers.Integral, np.integer)):
            max_samples = self.max_samples
        else:  # float
            max_samples = int(self.max_samples * X.shape[0])

        if not (0 < max_samples <= X.shape[0]):
            raise ValueError("max_samples must be in (0, n_samples]")

        if isinstance(self.max_features, (numbers.Integral, np.integer)):
            max_features = self.max_features
        else:  # float
            max_features = int(self.max_features * self.n_features_)

        if not (0 < max_features <= self.n_features_):
            raise ValueError("max_features must be in (0, n_features]")

        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available"
                             " if bootstrap=True")

        # Free allocated memory, if any
        self.estimators_ = None

        # Parallel loop
        n_jobs, n_estimators, starts = _partition_estimators(self.n_estimators,
                                                             self.n_jobs)
        seeds = random_state.randint(MAX_INT, size=self.n_estimators)

        all_results = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(_parallel_build_estimators)(
                n_estimators[i],
                self,
                X,
                y,
                sample_weight,
                seeds[starts[i]:starts[i + 1]],
                verbose=self.verbose)
            for i in range(n_jobs))

        # Reduce
        self.estimators_ = list(itertools.chain.from_iterable(
            t[0] for t in all_results))
        self.estimators_samples_ = list(itertools.chain.from_iterable(
            t[1] for t in all_results))
        self.estimators_features_ = list(itertools.chain.from_iterable(
            t[2] for t in all_results))

        if self.oob_score:
            self._set_oob_score(X, y)

        return self

def _parallel_build_estimators(n_estimators, ensemble, all_X, all_y, sample_weight,
                               seeds, verbose):
    """Private function used to build a batch of estimators within a job."""

    positives = np.where(all_y == 1)[0]
    unlabeled = np.where(all_y == 0)[0]
    
    X_positives = all_X[positives]
    X_unlabeled = all_X[unlabeled]
    y_positives = all_y[positives]
    y_unlabeled = all_y[unlabeled]

    # Retrieve settings
    n_samples, n_features = X_unlabeled.shape
    max_samples = ensemble.max_samples
    max_features = ensemble.max_features

    if (not isinstance(max_samples, (numbers.Integral, np.integer)) and
            (0.0 < max_samples <= 1.0)):
        max_samples = int(max_samples * n_samples)

    if (not isinstance(max_features, (numbers.Integral, np.integer)) and
            (0.0 < max_features <= 1.0)):
        max_features = int(max_features * n_features)

    bootstrap = ensemble.bootstrap
    bootstrap_features = ensemble.bootstrap_features
    
        #can't currently support sample weights
    if sample_weight is not None:
        raise ValueError("Can't currently support sample weight with PUBagging")

    support_sample_weight = False
    #support_sample_weight = has_fit_parameter(ensemble.base_estimator_,
     #                                         "sample_weight")
    #if not support_sample_weight and sample_weight is not None:
     #   raise ValueError("The base estimator doesn't support sample weight")

    # Build estimators
    estimators = []
    estimators_samples = []
    estimators_features = []

    for i in range(n_estimators):
        if verbose > 1:
            print("building estimator %d of %d" % (i + 1, n_estimators))

        random_state = check_random_state(seeds[i])
        seed = check_random_state(random_state.randint(MAX_INT))
        estimator = ensemble._make_estimator(append=False)

        try:  # Not all estimator accept a random_state
            estimator.set_params(random_state=seed)
        except ValueError:
            pass

        # Draw features
        if bootstrap_features:
            features = random_state.randint(0, n_features, max_features)
        else:
            features = sample_without_replacement(n_features,
                                                  max_features,
                                                  random_state=random_state)

        # Draw samples, using sample weights, and then fit
        if support_sample_weight:
            if sample_weight is None:
                curr_sample_weight = np.ones((n_samples,))
            else:
                curr_sample_weight = sample_weight.copy()

            if bootstrap:
                indices = random_state.randint(0, n_samples, max_samples)
                sample_counts = bincount(indices, minlength=n_samples)
                curr_sample_weight *= sample_counts

            else:
                not_indices = sample_without_replacement(
                    n_samples,
                    n_samples - max_samples,
                    random_state=random_state)

                curr_sample_weight[not_indices] = 0

            estimator.fit(all_X[:, features], all_y, sample_weight=curr_sample_weight)
            samples = curr_sample_weight > 0.

        # Draw samples, using a mask, and then fit
        else:
            if bootstrap:
                indices = random_state.randint(0, n_samples, max_samples)
            else:
                indices = sample_without_replacement(n_samples,
                                                     max_samples,
                                                     random_state=random_state)

            sample_counts = bincount(indices, minlength=n_samples)

            new_X=np.vstack((X_positives, X_unlabeled[indices]))
            new_y=np.concatenate((y_positives, y_unlabeled[indices]))

            estimator.fit(new_X[:, features], new_y)
            samples = sample_counts > 0.

        estimators.append(estimator)
        estimators_samples.append(samples)
        estimators_features.append(features)

    return estimators, estimators_samples, estimators_features

class SVC_mod(sklearn.svm.SVC, BaseEstimator, _LearntSelectorMixin):
    """
    A hacky way of getting SVC to support fit_transform?
    """
    


