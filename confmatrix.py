import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import os


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def compute_conf_matrix(y_test, y_pred, out_path, clf_name):
    # Compute confusion matrix
    cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    print('Confusion matrix')
    print(cm)

    plt.figure()
    plot_confusion_matrix(cm)
    plt.savefig(os.path.join(out_path, 'confmatrix__{}'.format(clf_name)))

    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('Normalized confusion matrix')
    print(cm_normalized)
    plt.figure()
    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
    plt.savefig(os.path.join(out_path, 'confmatrix_norm__{}'.format(clf_name)))
    # plt.show()
