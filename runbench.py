""" The main runner script for the Clairvoya benchmark code

[desc]

## Syntax

We use the sklearn syntax of

clf = classifier
clf_name = classifier name
X = features from dataset
y = labels from dataset
y_pred = predicted labels on test set

X and y are further defined

    X_<test/train>_<split_num>

We split the dataset into a test and training set and we do
this multiple times, each time has a different split number

eg X_train_0 and y_test_4
"""

import os

import voya_config
import datasetup
import benchmarks

out_path = voya_config.config['out_path']
if not os.path.isdir(out_path):
    os.makedirs(out_path)

y, X = datasetup.load_data(voya_config.config['data_file'])

data_splits = datasetup.split_train_data(y, X)

# TODO loop over split num only needed for cross-validation?
results_table_rows = []  # each row is a dict with column_name: value
for split_num, (X_train, y_train, X_test, y_test) in enumerate(data_splits):

    for clf_name, clf in voya_config.classifiers.iteritems():

        clf_name_split_num = '{} {}'.format(clf_name, split_num)
        print("Running {} sample {}".format(clf_name_split_num, split_num))

        clf.fit(X_train, y_train)
        y_pred = clf.predict_proba(X_test)[:, 1]

        bench_results = benchmarks.all_benchmarks(y_test, y_pred, clf_name_split_num, out_path)

        results_table_rows.append(bench_results)


results_table = benchmarks.results_dict_to_data_frame(results_table_rows)
print results_table