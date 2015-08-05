import matplotlib.pyplot as plt
import sklearn.calibration


def reliability_curve(prob_pos, test_labels, clf_name):
    """
    Adapted from http://scikit-learn.org/stable/auto_examples/calibration/plot_compare_calibration.html
        Author: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
        License: BSD Style.
    """
    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=3)
    ax2 = plt.subplot2grid((3, 1), (1, 0))
    ax3 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    fraction_of_positives, mean_predicted_value = \
        sklearn.calibration.calibration_curve(test_labels, prob_pos, n_bins=5)

    ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
             label="%s" % (clf_name,))

    ax2.hist(prob_pos[test_labels.astype(bool)], range=(0, 1), bins=15, label=clf_name, histtype="step", lw=2)

    ax3.hist(prob_pos[-(test_labels.astype(bool))], range=(0, 1), bins=15, label=clf_name, histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value for Positives")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    ax3.set_xlabel("Mean predicted value for not positives")
    ax3.set_ylabel("Count")
    ax3.legend(loc="upper center", ncol=2)
