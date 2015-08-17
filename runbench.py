""" The main runner script for the Clairvoya benchmark code

Config should be specified in a python config file (see voya_config_example.py) and ideally stored in config i.e
    config/config_name.py
This would then be ran as
    runbench.py config.config_name
    runbench.py config/config_name.py

Usage:
  runbench.py [<config>] [-v <level>]

Options:
    config    config file location
    -v level  verbosity level 0 = quiet, 1 = info, 2 = debug

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
import logging

from sklearn.grid_search import GridSearchCV
import sklearn.cross_validation
import docopt

import datasetup
import benchmarks
import voya_plotter

voya_logger = logging.getLogger('clairvoya')


def run_benchmark(config, classifiers, classifiers_gridparameters):
    """ Runs the benchmark code, see voya_config_example for argument explanation
    """

    default_config = {
        "data_file": None,  # input data
        "out_path": None,
        "num_folds": 5,
        "test_size": 0.2,
        "num_cores": 1,
        "pu_learning": False,
        "pu_rand_samp_frac": False,
        "verbosity": 0,
    }

    default_config.update(config)
    config = default_config

    set_verbosity_level(config["verbosity"])

    voya_logger.info("Starting Benchmark")

    out_path = config['out_path']
    if out_path is not None:
        if not os.path.isdir(out_path):
            os.makedirs(out_path)

    voya_logger.info('loading data from: {}'.format(config['data_file']))
    df = datasetup.load_data(config['data_file'])

    y, X_unscaled = datasetup.split_df_labels_features(df)
    X = datasetup.scale_features(X_unscaled)

    if config["pu_learning"]:  # input of positive, negative and unlabeled labels (1, -1, 0)
        voya_logger.info("PU Learning Benchmark")
        df_test, df_train = datasetup.split_test_train_df_pu(df, config['test_size'],
                                                             config["pu_rand_samp_frac"])

        y_test, X_test = datasetup.split_df_labels_features(df_test)
        y_train, X_train = datasetup.split_df_labels_features(df_train)

        # TODO (ryan) scale features (all together?)
        X_train = datasetup.scale_features(X_train)
        X_test = datasetup.scale_features(X_test)

    else:  # input of positive and negative (i.e 1, 0)
        X_train, y_train, X_test, y_test = datasetup.get_stratifed_data(y, X, config['test_size'])

    results_table_rows = {}  # each row is a dict with column_name: value

    skf = sklearn.cross_validation.StratifiedKFold(y_train, n_folds=config['num_folds'])

    for clf_name, clf_notoptimized in classifiers.iteritems():
        voya_logger.info("Running {}".format(clf_name))

        clf_results = {'clf_name': clf_name}

        param_grid = classifiers_gridparameters[clf_name]

        if param_grid is None:
            voya_logger.info('Skipping grid search for {}'.format(clf_name))
            voya_logger.debug("clf_notoptimized {}".format(clf_notoptimized))

            clf_fitted = clf_notoptimized.fit(X_train, y_train)

        else:
            clf = GridSearchCV(estimator=clf_notoptimized, param_grid=param_grid, cv=skf, scoring='roc_auc')
            clf_fitted = clf.fit(X_train, y_train).best_estimator_
            clf_optimal_parameters = clf.best_params_
            clf_results['clf_optimal_parameters'] = clf_optimal_parameters
            voya_logger.info(clf_name, clf_optimal_parameters)

        voya_logger.debug('X = {}'.format(clf_fitted))

        y_pred = clf_fitted.predict_proba(X_test)[:, 1]
        
        y_pred_label = clf_fitted.predict(X_test)
        
        clf_results.update({
            'y_pred': y_pred,
            'y_pred_label' : y_pred_label,
            'clf': clf_fitted,
            'clf_notoptimized': clf_notoptimized,
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'param_grid': param_grid,
            'X': X,
            'y': y,
        })


        #plot random forest decision trees
        if clf_results['clf_name'] == "Random Forests":
            voya_plotter.plot_trees(clf_results['clf'],out_path)




        voya_logger.info("Benchmarking {}".format(clf_name))
        benchmarks.all_benchmarks(clf_results, out_path)

        if out_path is not None:
            voya_plotter.plot_boundary(X_train, y_train, clf_name, clf_notoptimized, out_path)
            voya_plotter.roc_curve_cv(X_train, y_train, clf_name, clf_notoptimized, param_grid, out_path)

        # Cross validation using ROC curves TODO (ryan) think about moving this into benchmarks

        results_table_rows[clf_name] = clf_results

    voya_logger.info("\n#######\nResults\n#######")
    num_positives_y_train = y_train.sum()
    voya_logger.info("Training: positives = {}, negatives/unlabelled={}".format(num_positives_y_train, len(y_train)-num_positives_y_train))
    num_positives_y_test = y_test.sum()
    voya_logger.info("Testing: positives = {}, negatives={}".format(num_positives_y_test, len(y_test)-num_positives_y_test))

    results_table = benchmarks.results_dict_to_data_frame(results_table_rows)
    voya_logger.info('\n{}'.format(results_table))

    return results_table_rows


def set_verbosity_level(level):
    """ Set the console verbosity level, 0 - silent, 1 - info, 2 - debug
    """

    levels = (logging.CRITICAL, logging.INFO, logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(levels[level])
    voya_logger.setLevel(levels[level])
    voya_logger.addHandler(console_handler)

    if level == 2:
        formatter = logging.Formatter('%(asctime)s:%(levelname)s: %(message)s', "%H:%M:%S")
    else:
        formatter = logging.Formatter('%(message)s', "%H:%M:%S")
    console_handler.setFormatter(formatter)


if __name__ == '__main__':
    arguments = docopt.docopt(__doc__)

    config_module_name = arguments['<config>']
    if config_module_name is None:  # use default
        config_module_name = 'voya_config_example_pu'
    else:
        config_module_name = datasetup.parse_config_module_name(config_module_name)

    # This config loading as a module may not be entirely sensible, but is very quick for prototyping
    voya_config = importlib.import_module(config_module_name)

    if arguments["-v"] is not None:  # overwrite config verbosity
        voya_config.config["verbosity"] = int(arguments["-v"])

    results = run_benchmark(voya_config.config, voya_config.classifiers, voya_config.classifiers_gridparameters)
