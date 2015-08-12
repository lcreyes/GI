""" Config file Example for PU Learning

See description in voya_config_example for more information
"""

import sklearn.svm
import sklearn.ensemble

from puLearning.puAdapter import PUAdapter

config = {
    "data_file": "data/test_data_formatted.csv",
    "out_path": "output",
    "num_folds": 2,
    "test_size": 0.2,
    "pu_learning": True,
}

svc_estimator = sklearn.svm.SVC(C=10, kernel='rbf', gamma=0.4, probability=True)


classifiers = {
    'puestimator': PUAdapter(svc_estimator, hold_out_ratio=0.2),
    'Bagging SVC': sklearn.ensemble.BaggingClassifier(svc_estimator),
}

classifiers_gridparameters = {
   'puestimator': None,
   'Bagging SVC': {'n_estimators': [50, 100], 'max_samples': [0.1, 0.3]},
}
