""" The main runner script for the Clairvoya benchmark code

Config should be specified in a python config file (see voya_config_example.py) and ideally stored in config i.e
    config/config_name.py
This would then be ran as
    runbench.py config.config_name
    runbench.py config/config_name.py

Usage:
  runbench.py [<config>]

Notes:
    Currently the config files are stored in python, these aren't very portable and are not the best end solution
    but are much faster for prototyping new classifiers. If these become cumbersome we should rethink as they arent a
    great solution.

Syntax:
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
import importlib

from sklearn.grid_search import GridSearchCV
import sklearn.cross_validation

import datasetup
import benchmarks
import docopt


if not __name__ == '__main__':
    exit('Script is not importable')

arguments = docopt.docopt(__doc__)

config_module_name = arguments['<config>']
if config_module_name is None:  # use default
    config_module_name = 'voya_config_example'
else:
    config_module_name = datasetup.parse_config_module_name(config_module_name)

# This config loading as a module may not be entirely sensible, but is very quick for prototyping
voya_config = importlib.import_module(config_module_name)
print 'config file: {}.py'.format(config_module_name)

out_path = voya_config.config['out_path']
if not os.path.isdir(out_path):
    os.makedirs(out_path)

y, X = datasetup.load_data(voya_config.config['data_file'])

X_train, y_train, X_test, y_test = datasetup.get_stratifed_data(y, X, voya_config.config['test_size'])

# TODO loop over split num only needed for cross-validation?
results_table_rows = []  # each row is a dict with column_name: value

skf = sklearn.cross_validation.StratifiedKFold(y_train, n_folds=voya_config.config['num_folds'])

for clf_name, clf_notoptimized in voya_config.classifiers.iteritems():
    print("Running {}".format(clf_name))
    param_grid = voya_config.classifiers_gridparameters[clf_name]

    if param_grid is None:
        print 'Skipping grid search for {}'.format(clf_name)
        print "clf_notoptimized {}".format(clf_notoptimized)

        clf_fitted = clf_notoptimized.fit(X_train, y_train)

    else:
        clf = GridSearchCV(estimator=clf_notoptimized, param_grid=param_grid, cv=skf, scoring='roc_auc')
        clf_fitted = clf.fit(X_train, y_train).best_estimator_
        clf_optimalParameters = clf.best_params_
        print (clf_name, clf_optimalParameters)

    print 'X = ', clf_fitted

    y_pred = clf_fitted.predict_proba(X_test)[:, 1]

    print("Benchmarking {}".format(clf_name))
    bench_results = benchmarks.all_benchmarks(y_test, y_pred, clf_name, out_path)
    results_table_rows.append(bench_results)


results_table = benchmarks.results_dict_to_data_frame(results_table_rows)
print results_table
