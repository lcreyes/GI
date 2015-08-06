import matplotlib.pyplot as plt
import sklearn.calibration
import sklearn.preprocessing
import numpy
import seaborn


def reliability_curve(y_test, y_pred, clf_name):
    """
    Adapted from http://scikit-learn.org/stable/auto_examples/calibration/plot_compare_calibration.html
        Author: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
        License: BSD Style.
    """
    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0))
    ax2 = plt.subplot2grid((3, 1), (1, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    fraction_of_positives, mean_predicted_value = \
        sklearn.calibration.calibration_curve(y_test, y_pred, n_bins=5)

    ax1.set_title('Calibration plots  (reliability curve)')
    ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
             label="%s" % (clf_name,))
    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")

    ax2.set_xlabel("Mean predicted value")
    # for Positives
    ax2.hist(y_pred[y_test.astype(bool)], range=(0, 1), bins=15, label="Positives",
            histtype="stepfilled", lw=2, color='b', alpha=0.3, normed=1)
    ax2.set_ylabel("Normalised count")
    ax2.legend(loc="upper center", ncol=2)

    # for Negatives
    ax2.hist(y_pred[-(y_test.astype(bool))], range=(0, 1), bins=15, label="Negatives",
            histtype="stepfilled", lw=2, color='r', alpha=0.3, normed=1)
    ax2.legend(loc="upper center", ncol=2)


def confusion_matrix(y_test, y_pred, clf_name, threshold=0.5):
    """ Generates the plot for the confusin matrix

    Also does (which maybe it shouldnt)
    * Binarizer step (probabilities to 0 and 1) which should really be done and decided elsewhere
    * confusion matrix generation

    :param y_test:
    :param y_pred:
    :param clf_name:
    :param threshold:
    :return:
    """

    # TODO may want to move cm generation code out of here if we also want numeric output
    binarizer = sklearn.preprocessing.Binarizer(threshold)
    y_pred_binary = binarizer.transform(y_pred)[0]

    plt.figure(figsize=(10, 10))
    cmap = plt.cm.Blues

    plt.title("{} Normalised Confusion Matrix".format(clf_name))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    cm = sklearn.metrics.confusion_matrix(y_pred_binary, y_test)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
    plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
    plt.colorbar()
