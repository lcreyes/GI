""" Config file Example for searching with PU Learning

See description in voya_config_example for more information
"""

import sklearn.svm
import sklearn.linear_model
import sklearn.ensemble
import sklearn.pipeline

import pulearning

config = {
    'voya_mode': 'pusearch',
    "data_file": "data/test1_uni_f.csv",
    "out_path": None,
    "num_folds": 1,
    "test_size": 0.2,
    "pu_learning": True,
    "num_cores": 3,  # -1 means num_cores available

    "verbosity": 1,
    "random_forest_tree_plot": False,
    "auc_folds": 5,
    'search_results_file': 'test1_search_results.csv',  # csv file that records the results of each run
    'soft_search_run': False, #  if True builds on the previous results, if false overwrites the results file
    'search_range': (0.5, 1, 2),  # range of values to run over
    'runs_per_search': 1,  # number of times to run the search per parameter per classifier
    'search_live_plot': False,  # whether the show the progress as a plot
    'constant_test_train': True,  # otherwise will resplit every run_per_search
}

rf_estimator = sklearn.ensemble.RandomForestClassifier(max_depth=7, n_estimators=70, n_jobs=config["num_cores"])

RFBagging = sklearn.pipeline.Pipeline([
    ('rf', rf_estimator),
    ('po', pulearning.PUBagging(rf_estimator, max_samples=0.1, n_estimators=100, n_jobs=config["num_cores"])),
])

RFDoubleWeight = sklearn.pipeline.Pipeline([
    ('rf', rf_estimator),
    ('dw', pulearning.PULearnByDoubleWeighting(rf_estimator)),
])


classifiers = {
    'RF_Bagging': RFBagging,
    'RF_DoubleWeight': pulearning.PULearnByDoubleWeighting(rf_estimator),
    'RF': rf_estimator,
}

classifiers_gridparameters = {
    'RF_Bagging': None,
    'RF_DoubleWeight': None,
    'RF': None,

    # 'RF_DoubleWeight(E&N2008)': {"rf__n_estimators": [70], 'rf__max_depth': [7]},
    # 'RF_Bagging': {'po__n_estimators': [100], 'po__max_samples': [0.01, 0.03, 0.05, 0.1, 0.3, 0.5],
    #                "rf__n_estimators": [10, 30, 50, 70, 100], 'rf__max_depth': [1, 2, 3, 4, 5, 7, 10]},
}
