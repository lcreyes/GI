from config import config, classifiers
from datasetup import load_data, split_train_data
from benchmarks import all_benchmarks

import os

out_path = config['out_path']
if not os.path.isdir(out_path):
    os.makedirs(out_path)

# df = load_data(config['data_file'])

# TODO turn df into train_labels, features and test

labels, features = load_data(config['data_file'])

stratifiedShuffledData = split_train_data(labels)

for train_index, test_index in stratifiedShuffledData:
    train_features = features[train_index]
    train_labels = labels[train_index]
    test_features = features[test_index]
    test_labels = labels[test_index]

    for clf_name, clf in classifiers.iteritems():
        print("Running {}".format(clf_name))
        clf.fit(train_features, train_labels)
        
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(test_features)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(test_features)

        all_benchmarks(prob_pos, test_labels, clf_name, out_path)