from config import config, classifiers
from datasetup import load_data, test_data

import os

out_path = config['out_path']
if not os.path.isdir(out_path):
    os.makedirs(out_path)

# df = load_data(config['data_file'])

# TODO turn df into train_labels, features and test

train_features, train_labels, test_features = test_data()

for clf_name, clf in classifiers.iteritems():
    print("Running {}".format(clf_name))
    clf.fit(train_features, train_labels)
    prob_pos = clf.predict_proba(test_features)

    print prob_pos