""" Config file Example

This is an example config file which you SHOULDNT CHANGE EXCEPT TO WORK WITH NEW FEATURES. You should test this config
file works before committing and fix it if not. For general use, copy it elsewhere i.e. config/config_1.py which
isn't version controlled.

This file is also used for testing, and is therefore structured to cover the different paths through the code
"""

import sklearn.linear_model
import sklearn.neural_network
import sklearn.pipeline
import sklearn.ensemble

#### Basic configuration
config = {
    "data_file": "data/test_data_formatted.csv",  # input data
    "out_path": "output/norm/",  # output directory (plots/tables)
    "num_folds": 2,   # number of folds for K-Fold
    "test_size": 0.2,  # fraction of sample used as tex
    "num_cores": 3,
    "pu_learning": False,  # input dataset is PU (i.e contains positive 1, unlabeled 0, and negative -1 labels)
    # if True and pu_learning=true will randomly sample this proportion of unlabelled to be considered negative
    "pu_rand_samp_frac": False
}

#### Any custom code to combine new classifiers goes here

# Setup BernoulliRBM Neural net with logistic classifiers
logistic = sklearn.linear_model.LogisticRegression()
rbm = sklearn.neural_network.BernoulliRBM(verbose=True)
rbm_logistic = sklearn.pipeline.Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])

#### Specifiy SKLearn style classifiers here
classifiers = {
    'Logistic Regression': sklearn.linear_model.LogisticRegression(),
    'Random Forests': sklearn.ensemble.RandomForestClassifier(n_estimators=100, n_jobs=config["num_cores"]),
    # 'SVC': sklearn.svm.SVC(C=1.0, probability=True),
    # 'rbm_logistic': rbm_logistic,
    'Gradient Boosting': sklearn.ensemble.GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=2),
}

#### Specifiy arguments and parameters for grid search here, keys must match `classifiers` use None to use defaults
# given in initialisation
classifiers_gridparameters = {
    'Logistic Regression': {'C': [0.4, 0.8], 'penalty': ["l1"], 'class_weight': [None]},
    'Random Forests': None,  # dont grid search
    # 'Random Forests': {"max_depth": [3, None], "max_features": [1, 3, 10], "min_samples_split": [1, 3, 10],"min_samples_leaf": [1, 3, 10], "bootstrap": [True, False], "criterion": ["gini", "entropy"]},
    'SVC': [{'kernel': ['linear'], 'C': [0.2, 0.4, 0.6, 0.8, 1.0]}],
    'rbm_logistic': {'steps': [('rbm', rbm), ('logistic', logistic)]},
    'Gradient Boosting': {"n_estimators": [10, 50, 100], 'learning_rate': [0.1, 0.2, 0.3, 0.5], 'max_depth': [1, 2, 5],},
}
