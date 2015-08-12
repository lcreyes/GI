""" Config file Example for PU Learning

See description in voya_config_example for more information
"""

import sklearn.svm
import sklearn.linear_model
import sklearn.ensemble

from puLearning.puAdapter import PUAdapter
from puLearning.SVM_DoubleWeight import SVM_DoubleWeight

config = {
    "data_file": "data/test_Gaussians_formatted.csv",
    "out_path": "output",
    "num_folds": 2,
    "test_size": 0.2,
    "pu_learning": True,
}

#best parameters for rbf kernel(according to GridSearch)
#svc_estimator = sklearn.svm.SVC(C=2.5, kernel='rbf', gamma=0.2, class_weight='auto')
#best parameter for linear kernel (according to GridSearch)
svc_estimator = sklearn.svm.SVC(C=2.5, kernel='linear', class_weight='auto', probability=True)

#best parameters for LR (according to GridSearch)
LR_estimator = sklearn.linear_model.LogisticRegression(C=0.4, penalty='l1')


classifiers = {
    'puestimator': PUAdapter(svc_estimator, hold_out_ratio=0.2),
    # 'Bagging SVC': sklearn.ensemble.BaggingClassifier(svc_estimator),
    'Bagging LR': sklearn.ensemble.BaggingClassifier(LR_estimator),
    'SVM_DoubleWeight(E&N2008)': SVM_DoubleWeight(svc_estimator),
}

classifiers_gridparameters = {
   'puestimator': None,
   'Bagging SVC': {'n_estimators': [100, 200, 300], 'max_samples': [0.1, 0.3, 0.5, 0.7]},
   'Bagging LR': {'n_estimators':[100, 200, 300], 'max_samples':[0.1, 0.3, 0.5, 0.7]},
   'SVM_DoubleWeight(E&N2008)': None,


}
