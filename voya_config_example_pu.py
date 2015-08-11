""" Config file Example for PU Learning

This is an example config file which you SHOULDNT CHANGE EXCEPT TO WORK WITH NEW FEATURES. You should test this config
file works before committing and fix it if not. For general use, copy it elsewhere i.e. config/config_1.py which
isn't version controlled.
"""

import sklearn.svm

from puLearning.puAdapter import PUAdapter

config = {
    "data_file": "data/test_data_formatted.csv",
    "out_path": "output",
    "num_folds": 2,
    "test_size": 0.2,
    "pu_learning": True,
}

estimator = sklearn.svm.SVC(C=10, kernel='rbf', gamma=0.4, probability=True)

classifiers = {
    'puestimator': PUAdapter(estimator, hold_out_ratio=0.2)
}

classifiers_gridparameters = {
    'puestimator': None
}
