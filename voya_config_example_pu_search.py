""" Config file Example for searching with PU Learning

See description in voya_config_example for more information
"""

import sklearn.svm
import sklearn.linear_model
import sklearn.ensemble
import sklearn.pipeline
import sklearn.semi_supervised
import pulearning

config = {
    'voya_mode': 'pusearch',
    "data_file": "data/large_uni_f.csv",
 #   "data_file": "data/test1_uni_f.csv",
    "out_path": 'output/localAUC',
    "num_folds": 5,
    "test_size": 0.2,
    "pu_learning": True,
    "num_cores": 1,  # -1 means num_cores available

    "verbosity": 1,
    "random_forest_tree_plot": False,
    "auc_folds": 10,
    'search_results_file': 'search_results.csv',  # csv file that records the results of each run
    'soft_search_run': False, #  if True builds on the previous results, if false overwrites the results file
    #'search_range': (0.1, 1, 5, 10, 20),  # range of values to run overvalues to run over
    'search_range': (0.1, 1),  # range of values to run overvalues to run over
    'runs_per_search': 1,  # number of times to run the search per parameter per classifier
    'test_neg_to_pos_ratio': 1.,
    'ranking_Frac': 0.2,
    'constant_test_train': True,
    'includes_neg_inTrain': False,
}


# basic methods
rf_estimator = sklearn.ensemble.RandomForestClassifier()


#pipeline methods

RFBagging = sklearn.pipeline.Pipeline([
    ('rf', rf_estimator),
    ('po', pulearning.PUBagging(rf_estimator, max_samples=0.1, n_estimators=10, n_jobs=config["num_cores"])),
])

RFDoubleWeight = pulearning.PULearnByDoubleWeighting(rf_estimator)

RFPosOnly = sklearn.pipeline.Pipeline([
    ('rf' , rf_estimator),
    ('po', pulearning.PosOnly(rf_estimator)),
])



# best parameters for LR (according to GridSearch)


classifiers = {
    'RF_Bagging': RFBagging,
    'RF_DoubleWeight': pulearning.PULearnByDoubleWeighting(rf_estimator),
    'Random Forest': rf_estimator,
}

classifiers_gridparameters = {
    'Random Forest': {"n_estimators": [50, 100], "max_features": ["sqrt", None], "oob_score": [False],
                   "bootstrap": [True], "criterion": ["gini", "entropy"], "n_jobs": [1], "class_weight":[None, 'subsample']},
    'RF_Bagging': {'po__n_estimators': [100], 'po__max_samples': [0.3, 0.6, 0.9],  "po__n_jobs": [1],
                   "rf__max_features": ["sqrt", None], "rf__oob_score": [False],
                   "rf__bootstrap": [True], "rf__criterion": ["gini", "entropy"], "rf__n_jobs": [1], "rf__class_weight":[None, 'subsample']},
    'RF_DoubleWeight': {'estimator__bootstrap': [True],
                        'estimator__class_weight': [None],
                        'estimator__criterion': ['gini', 'entropy'],
                        'estimator__max_features': ['sqrt', None],
                        'estimator__n_estimators': [50, 100],
                        'estimator__n_jobs': [1],
                        'estimator__oob_score': [False]},
}

