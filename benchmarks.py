""" Handles more complex benchmarking code and output, probably the loop aswell
"""

import os.path
import logging

import matplotlib.pyplot as plt
import pandas
import sklearn.metrics

import voya_plotter
import numpy as np

voya_logger = logging.getLogger('clairvoya')

try:
    import optunity.metrics
    roc_pu_enabled = True
except ImportError:
    voya_logger.info('optunity not installed, disabled roc_pu plot')
    roc_pu_enabled = False


def all_benchmarks(clf_results, out_path, auc_folds=1):
    """ Runs all the benchmarks for given clf result saving the output to out_path

    Current classifier structure for plots is make a function in voya_plotter and call it here currently plots are
     saved here with plt.savefig and results are stored in results_dict in a 'benchmark': value, basis

    :param y_pred: predicted labels
    :param y_test: known labels
    :param clf_name: name of the clf used
    :param out_path: output plot/file save path
    :return:
    """

    y_test = clf_results["y_test"]
    y_pred = clf_results["y_pred"]
    y_pred_label = clf_results ["y_pred_label"]
    clf_name = clf_results["clf_name"].replace(' ', '')  # no spaces in plot file name
    clf = clf_results["clf"]
    X_test = clf_results["X_test"]

    if auc_folds > 1:
        scores = sklearn.cross_validation.cross_val_score(clf, X_test, y_test, cv=auc_folds, scoring='roc_auc')
        clf_results['pretty_auc_score'] = "%0.2f(+/-%0.2f)" % (scores.mean(), scores.std()/np.sqrt(auc_folds))
        clf_results['auc_score'] = scores.mean()
        clf_results['auc_std_err'] = scores.std()/np.sqrt(auc_folds)
    else:
        clf_results['auc_score'] = sklearn.metrics.roc_auc_score(y_test, y_pred)




    clf_results['f1_score'] = sklearn.metrics.f1_score(y_test, y_pred_label)

    if out_path is not None:  # output plots to out_path
        voya_logger.debug('Generating Reliability Curve plot')
        voya_plotter.reliability_curve(clf_results)
        plt.savefig(os.path.join(out_path, 'reli_curve__{}'.format(clf_name)), bbox_inches='tight')

        voya_logger.debug('Generating Confusion Matrix plot')
        voya_plotter.confusion_matrix(clf_results)
        plt.savefig(os.path.join(out_path, 'conf_matrix__{}'.format(clf_name)), bbox_inches='tight')

        voya_logger.debug('Generating roc curve plot')
        voya_plotter.roc_curve(clf_results)
        plt.savefig(os.path.join(out_path, 'roc__{}'.format(clf_name)), bbox_inches='tight')

        voya_logger.debug('Generating plot boundary plot')
        voya_plotter.plot_boundary(clf_results, runPCA=True)
        plt.savefig(os.path.join(out_path, 'boundary__{}'.format(clf_name)), bbox_inches='tight')

        voya_logger.debug('Generating roc curve cv')
        voya_plotter.roc_curve_cv(clf_results)
        plt.savefig(os.path.join(out_path, 'roc_cv__{}'.format(clf_name)), bbox_inches='tight')

        voya_logger.debug('Generating prVSranking curve plot')
        voya_plotter.prVSranking_curve(clf_results)
        plt.savefig(os.path.join(out_path, 'prVsRank__{}'.format(clf_name)), bbox_inches='tight')

        if roc_pu_enabled:
            voya_logger.debug('Generating PU ROC curve plot')
            voya_plotter.roc_pu(clf_results)
            plt.savefig(os.path.join(out_path, 'roc_pu__{}'.format(clf_name)), bbox_inches='tight')

        plt.close("all")  # perhaps a bad idea to put a close all here but all the plots will remain open otherwise


def results_dict_to_data_frame(results_dict):
    """ Takes input of a dict of dicts, each dict containing the same keys defined in `all_benchmarks`

    The only essential key is clf_name which forms the index. The rest are assumed to be classifiers which can be
    added on the fly in the all_benchmarks function.

    :param results_table_rows: {clf_1:{col1: val1, col2: val2}, clf2:{col1, val2, col2: val2}...}
    :return:
    """

    results_table_rows = [row for row in results_dict.values()]
    results_table = pandas.DataFrame(results_table_rows, columns=['clf_name', 'auc_score', 'f1_score']).sort('clf_name')
    results_table.rename(columns={'clf_name':'Classifier', 'auc_score': 'AUC Score', 'f1_score': 'F1 Score'}, inplace=True)

    return results_table
