""" Quick config file
"""

import sklearn.linear_model
import sklearn.neural_network
import sklearn.pipeline
import sklearn.ensemble

config = {
    "data_file": "test_data.csv",
    "out_path": "output",
    "num_folds":5,
    "test_size": 0.2,
    "random_seed": 49,  # currently only for shuffle
}

# Setup BernoulliRBM Neural net with logistic classifiers
logistic = sklearn.linear_model.LogisticRegression()
rbm = sklearn.neural_network.BernoulliRBM(random_state=0, verbose=True)
rbm_logistic = sklearn.pipeline.Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])

classifiers = {
    'lr': sklearn.linear_model.LogisticRegression(),
    'rfc': sklearn.ensemble.RandomForestClassifier(n_estimators=100),
    'svc': sklearn.svm.SVC(C=1.0, probability=True),
    'rbm_logistic': rbm_logistic
}
