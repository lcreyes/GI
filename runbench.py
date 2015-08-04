from config import config, classifiers
from datasetup import load_data, test_data
from benchmarks import all_benchmarks

import os

out_path = config['out_path']
if not os.path.isdir(out_path):
    os.makedirs(out_path)

# df = load_data(config['data_file'])

# TODO turn df into train_labels, features and test

train_features, train_labels, test_features, test_labels = test_data()

for clf_name, clf in classifiers.iteritems():
    print("Running {}".format(clf_name))
    clf.fit(train_features, train_labels)

    if hasattr(clf, "predict_proba"):
        prob_pos = clf.predict_proba(test_features)[:, 1]
    else:  # use decision function
        prob_pos = clf.decision_function(test_features)

    all_benchmarks(prob_pos, test_labels, clf_name, out_path)