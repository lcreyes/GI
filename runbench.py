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
from sklearn.grid_search import GridSearchCV
import sklearn.cross_validation

out_path = voya_config.config['out_path']
if not os.path.isdir(out_path):
    os.makedirs(out_path)

y, X = datasetup.load_data(voya_config.config['data_file'])

y_train = y_test = y
X_train = X_test = X

# TODO loop over split num only needed for cross-validation?
results_table_rows = []  # each row is a dict with column_name: value

skf = sklearn.cross_validation.StratifiedKFold(y_train, n_folds=voya_config.config['num_folds'])

for clf_name, clf_notoptimized in voya_config.classifiers.iteritems():
    print("Running {}".format(clf_name))
    clf = GridSearchCV(estimator=clf_notoptimized, param_grid=voya_config.classifiers_gridparameters[clf_name],
                       cv=skf)

    # TODO should we give it the whole set or not?
    clf_optimized = clf.fit(X_train, y_train).best_estimator_
    y_pred = clf_optimized.predict_proba(X_test)[:, 1]

    print("Benchmarking {}".format(clf_name))
    bench_results = benchmarks.all_benchmarks(y_test, y_pred, clf_name, out_path)
    results_table_rows.append(bench_results)


results_table = benchmarks.results_dict_to_data_frame(results_table_rows)
print results_table
