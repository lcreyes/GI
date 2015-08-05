import os

import config
import datasetup
import benchmarks

out_path = config.config['out_path']
if not os.path.isdir(out_path):
    os.makedirs(out_path)

# df = load_data(config['data_file'])

# TODO turn df into train_labels, features and test

labels, features = datasetup.load_data(config.config['data_file'])

stratified_shuffled_data = datasetup.split_train_data(labels)

for train_index, test_index in stratified_shuffled_data:
    train_features = features[train_index]
    train_labels = labels[train_index]
    test_features = features[test_index]
    test_labels = labels[test_index]

    for clf_name, clf in config.classifiers.iteritems():
        print("Running {}".format(clf_name))
        clf.fit(train_features, train_labels)
        
        prob_pos = clf.predict_proba(test_features)[:, 1]

        benchmarks.all_benchmarks(prob_pos, test_labels, clf_name, out_path)