""" We use the Sklean syntax of

clf = classifier
X =
y =
"""

import os

import voya_config
import datasetup
import benchmarks

out_path = voya_config.config['out_path']
if not os.path.isdir(out_path):
    os.makedirs(out_path)

# TODO turn df into train_labels, features and test

labels, features = datasetup.load_data(voya_config.config['data_file'])

stratified_shuffled_data = datasetup.split_train_data(labels)

# TODO move seperation step out of loop and iterate over steps
# TODO functionalise
for i, (train_index, test_index) in enumerate(stratified_shuffled_data):
    train_features = features[train_index]
    train_labels = labels[train_index]
    test_features = features[test_index]
    test_labels = labels[test_index]

    for clf_name, clf in voya_config.classifiers.iteritems():
        clf_i_name = '{} {}'.format(clf_name, i)
        print("Running {} sample {}".format(clf_i_name, i))
        clf.fit(train_features, train_labels)
        
        prob_pos = clf.predict_proba(test_features)[:, 1]  # TODO standardise this variable

        benchmarks.all_benchmarks(prob_pos, test_labels, clf_i_name, out_path)