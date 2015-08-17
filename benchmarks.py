""" Handles more complex benchmarking code and output, probably the loop aswell
"""

import os.path
import logging

import matplotlib.pyplot as plt
import pandas
import sklearn.metrics

import voya_plotter

voya_logger = logging.getLogger('clairvoya')


def all_benchmarks(clf_results, out_path):
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

        voya_logger.debug('Generating tprVSranking curve plot')
        voya_plotter.tprVSranking_curve(clf_results)
        plt.savefig(os.path.join(out_path, 'tprVsRank__{}'.format(clf_name)), bbox_inches='tight')

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
