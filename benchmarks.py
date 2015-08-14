""" Handles more complex benchmarking code and output, probably the loop aswell
"""

import os.path
import logging

import matplotlib.pyplot as plt
import pandas

import voya_plotter

voya_logger = logging.getLogger('clairvoya')


def all_benchmarks(y_test, y_pred, clf_name, out_path):
    """ Runs all the benchmarks for given clf result saving the output to out_path

    Current classifier structure for plots is make a function in voya_plotter and call it here currently plots are
     saved here with plt.savefig and results are stored in results_dict in a 'benchmark': value, basis

    :param y_pred: predicted labels
    :param y_test: known labels
    :param clf_name: name of the clf used
    :param out_path: output plot/file save path
    :return:
    """

    results_dict = {'clf_name': clf_name}  # a row of the (eventual) results table

    # TODO (ryan) think about structure here, have a common interface and move the params to config like classifiers?
    voya_plotter.reliability_curve(y_test, y_pred, clf_name)
    plt.savefig(os.path.join(out_path, 'reli_curve__{}'.format(clf_name.replace(' ', ''))), bbox_inches = 'tight')

    voya_plotter.confusion_matrix(y_test, y_pred, clf_name)
    plt.savefig(os.path.join(out_path, 'conf_matrix__{}'.format(clf_name.replace(' ', ''))), bbox_inches = 'tight')

    # TODO (ryan) auc score should probably be computed here and passed to roc_curve, expecailly if we k-fold it in future
    auc_score = voya_plotter.roc_curve(y_test, y_pred, clf_name)
    plt.savefig(os.path.join(out_path, 'roc__{}'.format(clf_name.replace(' ', ''))), bbox_inches = 'tight')

    # auc_score = sklearn.metrics.roc_auc_score(y_test, y_pred)
    results_dict['auc_score'] = auc_score

    plt.close("all")  # perhaps a bad idea to put a close all here but all the plots will remain open otherwise

    return results_dict


def results_dict_to_data_frame(results_dict):
    """ Takes input of a dict of dicts, each dict containing the same keys defined in `all_benchmarks`

    The only essential key is clf_name which forms the index. The rest are assumed to be classifiers which can be
    added on the fly in the all_benchmarks function.

    :param results_table_rows: {clf_1:{col1: val1, col2: val2}, clf2:{col1, val2, col2: val2}...}
    :return:
    """

    results_table_rows = [row for row in results_dict.values()]

    results_table = pandas.DataFrame(results_table_rows, columns=['clf_name', 'auc_score']).sort('clf_name')
    results_table.rename(columns={'clf_name':'Classifier', 'auc_score': 'AUC Score'}, inplace=True)

    return results_table
