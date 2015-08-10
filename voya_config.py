""" Quick config file
"""

import sklearn.linear_model
import sklearn.neural_network
import sklearn.pipeline
import sklearn.ensemble

config = {
    "data_file": "data/test_data_formatted.csv",
    "out_path": "output",
    "num_folds": 2,
    "test_size": 0.2,
}

# Setup BernoulliRBM Neural net with logistic classifiers
logistic = sklearn.linear_model.LogisticRegression()
rbm = sklearn.neural_network.BernoulliRBM(verbose=True)
rbm_logistic = sklearn.pipeline.Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])

classifiers = {
    'lr': sklearn.linear_model.LogisticRegression(),
    'rfc': sklearn.ensemble.RandomForestClassifier(n_estimators=100),
    'svc': sklearn.svm.SVC(C=1.0, probability=True),
    # 'rbm_logistic': rbm_logistic,
    'gbc': sklearn.ensemble.GradientBoostingClassifier(),
}

classifiers_gridparameters = {
    'lr': {'C': [1, 10, 100, 1000], 'penalty': ["l1", "l2"], 'class_weight': [None, "auto"]},
    'rfc': {"max_depth": [3, None], "max_features": [1, 3, 10], "min_samples_split": [1, 3, 10],
            "min_samples_leaf": [1, 3, 10], "bootstrap": [True, False], "criterion": ["gini", "entropy"]},
    'svc': [{'kernel': ['linear'], 'C': [1, 10, 100]}],
    # 'rbm_logistic' : {'steps' : [('rbm', rbm), ('logistic', logistic)]},
    'gbc': {"n_estimators": [10, 100, 1000], 'learning_rate': [0.2, 0.5, 0.7, 1.0], 'max_depth': [1, 3, 10],}
}
