from config import config, classifiers
from datasetup import load_data

df = load_data(config.data_loc)

# TODO turn df into train_labels, features and test

for clf in classifiers:
    clf.fit(train_labels, train_features)
    clf.predict_proba(test_features)