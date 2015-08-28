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
import csv

from sklearn.grid_search import GridSearchCV
import sklearn.cross_validation
import sklearn.ensemble
import docopt
import matplotlib.pyplot as plt

import datasetup
import benchmarks
import voya_plotter

voya_logger = logging.getLogger('clairvoya')


def run_benchmark(config, classifiers, classifiers_gridparameters):
    """ Runs the benchmark code, see voya_config_example for argument explanation
    """

    default_config = {
        "data_file": None,  # input data
        "test_df": None,  # instead of data_file, give split data
        "train_df": None,

        "out_path": None,
        "num_folds": 5,
        "test_size": 0.2,
        "num_cores": 1,
        "pu_learning": False,
        "pu_rand_samp_frac": False,
        "verbosity": 0,
        "random_forest_tree_plot": False,
        "auc_folds": 1,
        'u_to_p_ratio': False,
    }

    default_config.update(config)
    config = default_config

    set_verbosity_level(config["verbosity"])

    voya_logger.info("Starting Benchmark")

    out_path = config['out_path']
    if out_path is not None:
        if not os.path.isdir(out_path):
            os.makedirs(out_path)


    # If we are given the test / train sets explicitly
    test_df = config["test_df"]
    train_df = config["train_df"]
    if test_df is not None and train_df is not None:
        y_test, X_test = datasetup.split_df_labels_features(test_df)
        y_train, X_train = datasetup.split_df_labels_features(train_df)
    elif config["data_file"] is not None:  # or load all the data and auto split
        voya_logger.info('loading data from: {}'.format(config['data_file']))

        try:
            df = datasetup.load_data(config['data_file'])
        except IOError:  # file doesnt exist, try seeing is its a df instead
            df = config['data_file']

        voya_logger.info("Input data labels \n{}".format(df.label.value_counts()))

        try:
            datasetup.scale_dataframe_features(df)
        except TypeError:  # Got a string as the DF (after IOError)
            raise VoyaConfigError('data_file is not a valid path to a file or a Pandas DF, got {}'.format(df))

        if config["pu_learning"]:  # input of positive, negative and unlabeled labels (1, -1, 0)
            voya_logger.info("PU Learning Mode On")

            if config["u_to_p_ratio"]:
                df = datasetup.downsample_pu_df(df, config["u_to_p_ratio"])

            df_test, df_train = datasetup.split_test_train_df_pu(df, config['test_size'],)

            y_test, X_test = datasetup.split_df_labels_features(df_test)
            y_train, X_train = datasetup.split_df_labels_features(df_train)

        else:  # input of positive and negative (i.e 1, 0)

            X, y = datasetup.split_df_labels_features(df)
            X_train, y_train, X_test, y_test = datasetup.get_stratifed_data(y, X, config['test_size'])
    else:
        raise ValueError("You must give either `test_df` and `train_df` OR `data_file` in config")

    results_table_rows = {}  # each row is a dict with column_name: value

    for clf_name, clf_notoptimized in classifiers.iteritems():
        voya_logger.info("Running {}".format(clf_name))

        clf_results = {'clf_name': clf_name}

        param_grid = classifiers_gridparameters[clf_name]

        if param_grid is None:
            voya_logger.info('Skipping grid search for {}'.format(clf_name))
            voya_logger.debug("clf_notoptimized {}".format(clf_notoptimized))

            clf_fitted = clf_notoptimized.fit(X_train, y_train)

        else:
            voya_logger.info('Performing grid search for {}'.format(clf_name))
            skf = sklearn.cross_validation.StratifiedKFold(y_train, n_folds=config['num_folds'])

            clf = GridSearchCV(estimator=clf_notoptimized, param_grid=param_grid, cv=skf, scoring='roc_auc',
                               n_jobs=config['num_cores'])

            clf_fitted = clf.fit(X_train, y_train).best_estimator_
            clf_optimal_parameters = clf.best_params_
            clf_results['clf_optimal_parameters'] = clf_optimal_parameters
            voya_logger.info('Optimal parameters are {}'.format(clf_optimal_parameters))

        voya_logger.debug('X = {}'.format(clf_fitted))

        y_pred = clf_fitted.predict_proba(X_test)[:, 1]

        y_pred_label = clf_fitted.predict(X_test)

        clf_results.update({
            'y_pred': y_pred,
            'y_pred_label': y_pred_label,
            'clf': clf_fitted,
            'clf_notoptimized': clf_notoptimized,
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'param_grid': param_grid,
        })

        voya_logger.info("Benchmarking {}".format(clf_name))
        benchmarks.all_benchmarks(clf_results, out_path,
                                  config["auc_folds"])  # TODO (ryan) split this up now into benchmarks and plots?

        if out_path is not None:  # TODO (ryan) non conforming plots, move to benchmarks
            if config["random_forest_tree_plot"] and isinstance(clf_fitted, sklearn.ensemble.RandomForestClassifier):
                voya_logger.debug('Generating random forrest plot')
                # TODO (ryan) weve hardcoded '2' where the feature start several times, export to var?
                feature_names = [colname.replace('url/tech/', '').replace('cid/tech/', '') for colname in
                                 df.columns[2:]]
                voya_plotter.plot_trees(clf_results['clf'], feature_names)

        results_table_rows[clf_name] = clf_results

    voya_logger.info("\n#######\nResults\n#######")
    num_positives_y_train = y_train.sum()
    voya_logger.info("Training: positives = {}, negatives/unlabelled={}".format(num_positives_y_train,
                                                                                len(y_train) - num_positives_y_train))
    num_positives_y_test = y_test.sum()
    voya_logger.info(
        "Testing: positives = {}, negatives={}".format(num_positives_y_test, len(y_test) - num_positives_y_test))

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
    fh = logging.FileHandler('clairvoya.log')
    fh.setLevel(levels[level])
    voya_logger.addHandler(fh)

    if level == 2:
        formatter = logging.Formatter('%(asctime)s:%(levelname)s: %(message)s', "%H:%M:%S")
    else:
        formatter = logging.Formatter('%(message)s', "%H:%M:%S")
    console_handler.setFormatter(formatter)


def run_search_benchmark(config, classifiers, classifiers_gridparameters):
    """ This works like run_bench except it calls runbench multiple times varying the fraction of unlabelled in the
    sample.

    Currently PU learning only, varies the fraction of unlabelled in the classifier as a function of positive

    Search parameters are set in the config dictionary. See the code for required config (in addition to that in runbench).

    :param config:
    :param classifiers:
    :param classifiers_gridparameters:
    :return:
    """

    default_config = {
        "data_file": None,  # input data
        "test_df": None,  # instead of data_file, give split data
        "train_df": None,

        "out_path": None,
        "num_folds": 5,
        "test_size": 0.2,
        "num_cores": 3,
        "pu_learning": False,
        "pu_rand_samp_frac": False,
        "verbosity": 0,
        "random_forest_tree_plot": False,
        "auc_folds": 1,
        'u_to_p_ratio': False,

        'voya_mode': 'pusearch',
        'search_results_file': '',  # csv file that records the results of each run
        'soft_search_run': True,  #  if True builds on the previous results, if false overwrites the results file
        'search_range': (0.5, 1, 2),  # range of values to run over
        'runs_per_search': 3,  # number of times to run the search per parameter per classifier
        'search_live_plot': False,
        'constant_test_train': True,  # otherwise will resplit every run_per_search
        'test_neg_to_pos_ratio': None,
    }

    default_config.update(config)
    config = default_config

    if config['constant_test_train']:  # Split test / train so we have a constant testing set
        try:
            df = datasetup.load_data(config['data_file'])
        except IOError:  # file doesnt exist, try seeing is its a df instead
            df = config['data_file']

        df_test, df_train = datasetup.split_test_train_df_pu(df, config['test_size'], 
                                                             test_neg_to_pos_ratio=config['test_neg_to_pos_ratio'])

        config["test_df"] = df_test
        config["train_df"] = df_train
        config["data_file"] = None

        if not config['runs_per_search'] == 1:  # no point doing more if we have a constant test/train
            voya_logger.warning('Setting runs_per_search to 1 as constant_test_train is True, change auc_folds instead')
            config['runs_per_search'] = 1

    save_file = config['search_results_file']
    search_range = config['search_range']
    runs_per_search = config['runs_per_search']

    voya_logger.info('Starting search benchmark')

    if not os.path.exists(save_file) or not config['soft_search_run']:
        with open(save_file, 'wb') as f:
            if config['constant_test_train']:
                f.write('clf,auc,gamma,std,folds,stderr\n')
            else:
                f.write('clf,auc,gamma\n')

    fig = None

    for gamma_num, gamma in enumerate(search_range):  # gamma is a single value in the search range
        voya_logger.info('Running classifiers for gamma={} ({}/{})'.format(gamma, gamma_num + 1, len(search_range)))

        for i in xrange(runs_per_search):
            config.update({"u_to_p_ratio": gamma})

            if config['constant_test_train']:
                config["train_df"] = datasetup.downsample_pu_df(df_train, config["u_to_p_ratio"])

            results_dict = run_benchmark(config, classifiers, classifiers_gridparameters)

            # Output
            csv_output = []
            for clf_name in classifiers.keys():
                if config['constant_test_train']:
                    csv_row = (clf_name, results_dict[clf_name]['auc_score'], gamma,
                               results_dict[clf_name]['auc_std'], results_dict[clf_name]['auc_folds'],
                               results_dict[clf_name]['auc_std_err'])
                else:
                    csv_row = (clf_name, results_dict[clf_name]['auc_score'], gamma)

                csv_output.append(csv_row)

            with open(save_file, 'ab') as f:
                csv_f = csv.writer(f)
                csv_f.writerows(csv_output)

            if config['search_live_plot']:
                plt.clf()
                fig = voya_plotter.pu_search_result(save_file, fig)
                plt.draw()


class VoyaConfigError(Exception):
    pass


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

    try:
        voya_mode = voya_config.config['voya_mode']
    except KeyError:
        raise VoyaConfigError('voya_mode undefined in config, must be "bench" or "pusearch"')

    if voya_mode == 'bench':
        results = run_benchmark(voya_config.config, voya_config.classifiers, voya_config.classifiers_gridparameters)
    elif voya_mode == 'pusearch':
        results = run_search_benchmark(voya_config.config, voya_config.classifiers, voya_config.classifiers_gridparameters)
    else:
        raise VoyaConfigError('config must define voya_mode as "bench" or "pusearch" got {}'
                              ''.format(voya_config.config['voya_mode']))
