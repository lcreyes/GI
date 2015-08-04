""" Quick config file
"""

from sklearn.linear_model import LogisticRegression

config = {
    "data_loc": "test_data.csv"
}

classifiers = {
    'lr': LogisticRegression(),
    # 'rfc': RandomForestClassifier(n_estimators=100)
}