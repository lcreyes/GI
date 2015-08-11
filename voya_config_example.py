""" Config file Example

This is an example config file which you SHOULDNT CHANGE EXCEPT TO WORK WITH NEW FEATURES. You should test this config
file works before committing and fix it if not. For general use, copy it elsewhere i.e. config/config_1.py which
isn't version controlled.
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
    'Logistic Regression': sklearn.linear_model.LogisticRegression(),
    # 'Random Forests': sklearn.ensemble.RandomForestClassifier(n_estimators=100),
    # 'SVC': sklearn.svm.SVC(C=1.0, probability=True),
    # 'BernoulliRBM Logistic': rbm_logistic,
    'Gradient Boosting': sklearn.ensemble.GradientBoostingClassifier(),

}

classifiers_gridparameters = {
    'Logistic Regression': {'C': [0.2, 0.4, 0.6, 0.8, 1.0], 'penalty': ["l1", "l2"], 'class_weight': [None, "auto"]},
    'Random Forests': {"max_depth": [3, None], "max_features": [1, 3, None, 'auto'], "min_samples_split": [1, 2, 3],
            "min_samples_leaf": [1, 2, 3, 4], "bootstrap": [True, False], "criterion": ["gini", "entropy"]},
     'SVC': [{'kernel': ['linear', 'rbf'], 'C': [0.2, 0.4, 0.6, 0.8, 1.0]}],
    'BernoulliRBM Logistic' : None,
    'Gradient Boosting': {"n_estimators": [10, 50, 100], 'learning_rate': [0.1, 0.2, 0.3, 0.5], 'max_depth': [1, 2, 5],}
}
