""" Config file Example for PU Learning

See description in voya_config_example for more information
"""

import sklearn.svm
import sklearn.linear_model
import sklearn.ensemble
import sklearn.pipeline

import pulearning

config = {
    'voya_mode': 'bench',
    "data_file": "data/test1_uni_f.csv",
    "out_path": "output/pu/",
    "num_folds": 2,
    "test_size": 0.2,
    "pu_learning": True,
    "num_cores": -1,  # -1 means num_cores available
    # if True and pu_learning=true will randomly sample this proportion of unlabelled to be considered negative
    # otherwise we will use all the unlabelled data to train
    "verbosity": 1,
    "random_forest_tree_plot": False,
    "auc_folds": 1,
    'u_to_p_ratio': 2,  # If false, uses all data, if a number will sample a random proportion of unlabeled
}

# best parameters for rbf kernel(according to GridSearch)
# svc_estimator = sklearn.svm.SVC(C=2.5, kernel='rbf', gamma=0.2, class_weight='auto')
# best parameter for linear kernel (according to GridSearch)
svc_estimator = sklearn.svm.SVC(C=2.5, kernel='linear', class_weight='auto', probability=True)

rf_estimator = sklearn.ensemble.RandomForestClassifier(max_depth=7, n_estimators=70, n_jobs=config["num_cores"])

RFBagging = sklearn.pipeline.Pipeline([
    ('rf', rf_estimator),
    ('po', pulearning.PUBagging(rf_estimator, max_samples=0.1, n_estimators=100, n_jobs=config["num_cores"])),
])

RFDoubleWeight = sklearn.pipeline.Pipeline([
    ('rf', rf_estimator),
    ('dw', pulearning.PULearnByDoubleWeighting(rf_estimator)),
])

# best parameters for LR (according to GridSearch)
LR_estimator = sklearn.linear_model.LogisticRegression(C=0.4, penalty='l1')


classifiers = {
    #'PosOnly(E&N2008)': pulearning.PosOnly(svc_estimator, hold_out_ratio=0.2, ),
    # 'Bagging SVC': sklearn.ensemble.BaggingClassifier(svc_estimator, n_estimators=100, max_samples=0.3,
    #                                                   n_jobs=config["num_cores"]),
    'RF_Bagging': RFBagging,
    # 'RFDoubleWeight': pulearning.PULearnByDoubleWeighting(rf_estimator),
    # 'Bagging LR': sklearn.ensemble.BaggingClassifier(LR_estimator, n_jobs=config["num_cores"]),
    # 'SVM_DoubleWeight(E&N2008)': pulearning.PULearnByDoubleWeighting(svc_estimator),
    #  'Bagging LR': pulearning.PUBagging(LR_estimator, n_estimators=20, max_samples=0.5),
    #  'SVM_DoubleWeight(E&N2008)': pulearning.SVMDoubleWeight(svc_estimator),
}

classifiers_gridparameters = {
    'PosOnly(E&N2008)': None,
    'RF_DoubleWeight(E&N2008)': {"rf__n_estimators": [70], 'rf__max_depth': [7]},
    'RF_Bagging': {'po__n_estimators': [100], 'po__max_samples': [0.01, 0.03, 0.05, 0.1, 0.3, 0.5],
                   "rf__n_estimators": [10, 30, 50, 70, 100], 'rf__max_depth': [1, 2, 3, 4, 5, 7, 10]},
    'Bagging SVC': {'n_estimators': [30, 100], 'max_samples': [0.1, 0.3, 0.7]},
    'Bagging LR': {'n_estimators': [30, 100], 'max_samples': [0.1, 0.3, 0.7]},
    'SVM_DoubleWeight(E&N2008)': None,
}
