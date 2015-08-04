""" Quick config file
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

config = {
    "data_file": "test_data.csv",
    "out_path": "output",
}

# Setup BernoulliRBM Neural net with logistic classifier TODO move to custom classifiers module?
logistic = LogisticRegression()
rbm = BernoulliRBM(random_state=0, verbose=True)
rbm_logistic = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])

classifiers = {
    'lr': LogisticRegression(),
    'rfc': RandomForestClassifier(n_estimators=100),
    'svc': SVC(C=1.0, probability=True),
    'rbm_logistic': rbm_logistic
}