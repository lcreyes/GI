""" Quick config file
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

config = {
    "data_file": "test_data.csv",
    "out_path": "output",
}

classifiers = {
    'lr': LogisticRegression(),
    'rfc': RandomForestClassifier(n_estimators=100),
    'svc': SVC(C=1.0, probability=True)
}