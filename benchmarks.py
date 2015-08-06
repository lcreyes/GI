""" Handles more complex benchmarking code and output, probably the loop aswell
"""

import os.path

import matplotlib.pyplot as plt
import sklearn.metrics

import voya_plotter


def all_benchmarks(y_test, y_pred, clf_name, out_path):
    """ Runs all the benchmarks for given clf result saving the output to out_path

    :param y_pred: predicted labels
    :param y_test: known labels
    :param clf_name: name of the clf used
    :param out_path: output plot/file save path
    :return:
    """

    voya_plotter.reliability_curve(y_test, y_pred, clf_name)
    plt.savefig(os.path.join(out_path, 'reli_curve__{}'.format(clf_name)))

    voya_plotter.confusion_matrix(y_test, y_pred, clf_name)
    plt.savefig(os.path.join(out_path, 'conf_matrix__{}'.format(clf_name)))

    auc_score = sklearn.metrics.roc_auc_score(y_test, y_pred)
    print '{} AUC Score = {}'.format(clf_name, auc_score)
