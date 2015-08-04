""" Handles more complex benchmarking code and output, probably the loop aswell
"""

import matplotlib.pyplot as plt
from sklearn.metrics import auc
from plotter import reliability_curve
import os.path


def all_benchmarks(prob_pos, labels, clf_name, out_path):
    """ Runs all benchmarks for given clf result

    :param prob_pos: probabilities from clf algo
    :param labels: the real (known) labels
    :param clf_name: name of the clf used
    :param out_path: output plot/file save path
    :return:
    """
    reliability_curve(prob_pos, labels, clf_name)
    plt.savefig(os.path.join(out_path, 'reli_curve__{}'.format(clf_name)))

    auc_score = auc(prob_pos, labels, reorder=True)
    print '{} AUC Score = {}'.format(clf_name, auc_score)