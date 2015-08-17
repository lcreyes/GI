import logging
import os.path

import matplotlib.pyplot as plt
import sklearn.calibration
import sklearn.preprocessing
import sklearn.metrics
from sklearn.grid_search import GridSearchCV
import numpy as np
import seaborn

voya_logger = logging.getLogger('clairvoya')


# TODO (ryan) plot titles should include classifer name (or elsewhere on plot)
def reliability_curve(clf_results):
    """
    Adapted from http://scikit-learn.org/stable/auto_examples/calibration/plot_compare_calibration.html
        Author: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
        License: BSD Style.
    """

    y_test = clf_results["y_test"]
    clf_name = clf_results["clf_name"]
    y_pred = clf_results["y_pred"]

    seaborn.set_style("darkgrid")
    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0))
    ax2 = plt.subplot2grid((3, 1), (1, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    fraction_of_positives, mean_predicted_value = \
        sklearn.calibration.calibration_curve(y_test, y_pred, n_bins=5, normalize=True)

    ax1.set_title('Calibration plots  (reliability curve)')
    ax1.set_title('{} - Calibration plots  (reliability curve)'.format(clf_name))
    ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
             label="%s" % (clf_name,))
    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")

    ax2.set_xlabel("Mean predicted value")
    # for Positives
    ax2.hist(y_pred[y_test.astype(bool)], range=(0, 1), bins=15, label="Positives",
            histtype="step", lw=2, color='b', alpha=0.3, normed=1)
    ax2.set_ylabel("Normalised count")
    ax2.legend(loc="upper center", ncol=2)

    # for Negatives
    ax2.hist(y_pred[-(y_test.astype(bool))], range=(0, 1), bins=15, label="Negatives",
            histtype="step", lw=2, color='r', alpha=0.3, normed=1)
    ax2.legend(loc="upper center", ncol=2)


def confusion_matrix(clf_results, threshold=0.5):
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

    y_test = clf_results["y_test"]
    clf_name = clf_results["clf_name"]
    y_pred = clf_results["y_pred"]

    # TODO may want to move cm generation code out of here if we also want numeric output
    binarizer = sklearn.preprocessing.Binarizer(threshold)
    y_pred_binary = binarizer.transform(y_pred)[0]

    cm = sklearn.metrics.confusion_matrix(y_pred_binary, y_test)

    seaborn.set_style("white")
    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    cmap = plt.cm.Blues

    ax1.set_title("{} Confusion Matrix".format(clf_name))
    ax1.set_title("{} - Confusion Matrix".format(clf_name))
    ax1.set_ylabel('True label')
    ax1.set_xlabel('Predicted label')
    ax1.locator_params(nbins=4)

    confMatrix1 = ax1.imshow(cm, interpolation='nearest', cmap=cmap)

    #Display the values of the conf matrix on the plot
    cm_bbox = {'facecolor':'white', 'alpha':0.5, 'pad':10}
    for i in range(2):
        for j in range(2):
            ax1.text(i,j,"%d" %cm[i,j], size=14, ha='center', bbox=cm_bbox)

    plt.colorbar(confMatrix1, ax=ax1)

    ax2.set_title("Normalised Confusion Matrix")
    ax2.set_ylabel('True label')
    ax2.set_xlabel('Predicted label')

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    confMatrix2 = ax2.imshow(cm_normalized, interpolation='nearest', cmap=cmap)

    for i in range(2):
        for j in range(2):
            ax2.text(i,j,"%4.2f" %cm_normalized[i,j], ha='center', size=14, bbox=cm_bbox)

    plt.colorbar(confMatrix2, ax=ax2)


def roc_curve(clf_results):
    """
    Adapted from http://scikit-learn.org/stable/auto_examples/calibration/plot_compare_calibration.html
        Author: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
        License: BSD Style.
    """

    y_test = clf_results["y_test"]
    clf_name = clf_results["clf_name"]
    y_pred = clf_results["y_pred"]

    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = sklearn.metrics.roc_curve(y_test, y_pred)
    roc_auc = sklearn.metrics.auc(fpr, tpr)

    # Plot of a ROC curve for a specific class
    seaborn.set_style("whitegrid")
    plt.figure(figsize=(7, 7))
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' %roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('{} - Receiver operating characteristic'.format(clf_name))
    plt.legend(loc="lower right")
    return roc_auc


def roc_curve_cv(X, y, clf_name, clf_notoptimized, param_grid, out_path):
    """
    Adapted from http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
    """
    ###############################################################################
    # Run classifier with cross-validation and plot ROC curves

    # Do an initial kfold on the data- n_folds will be the number of roc curves shown
    # Can be the same from voya_config
    cv = sklearn.cross_validation.StratifiedKFold(y, n_folds=6)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []

    seaborn.set_style("whitegrid")
    plt.figure(figsize=(7, 7))

    #Loop over each fold, make distinction on whether grid_search is on/off
    for i, (train, test) in enumerate(cv):
        if param_grid is None:
            ### Temporary solution while SVM_DoubleWeight(E&N2008) is fixed
            if(clf_name == 'SVM_DoubleWeight(E&N2008)'):
                clf_notoptimized.weights_available = False
            ###
            clf_fitted = clf_notoptimized.fit(X[train], y[train])

        else:
            #Need a second k-fold here to do Grid Sarch on this particular fold
            skf = sklearn.cross_validation.StratifiedKFold(y[train], n_folds=2)
            clf = GridSearchCV(estimator=clf_notoptimized, param_grid=param_grid, cv=skf, scoring='roc_auc')
            clf_fitted = clf.fit(X[train], y[train]).best_estimator_

        y_pred = clf_fitted.predict_proba(X[test])[:, 1]

        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y[test], y_pred)
        mean_tpr += np.interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = sklearn.metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6)) #, label='Luck')

    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = sklearn.metrics.auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',
        label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

    plt.xlim([0., 1.])
    plt.ylim([0., 1.])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('%s - Receiver operating characteristic CV' %clf_name)
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(out_path, 'roc_cv__{}'.format(clf_name.replace(' ', ''))), bbox_inches = 'tight')


def plot_boundary(X_all, y, clf_name, clf_notoptimized, out_path):
    #print "Attempting to plot decision boundary..."
    #take the first two features TODO: implement PCA
    X = X_all[:, :2]

    # create a mesh to plot in
    h = .01  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

    seaborn.set_style("whitegrid")
    plt.figure(figsize=(7, 7))

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].

    #plt.subplot(2, 2, i + 1)
    #plt.subplots_adjust(wspace=0.4, hspace=0.4)
    clf_fitted = clf_notoptimized.fit(X, y)

    Z = clf_fitted.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,1]
    Z = np.clip(Z, 0., 1.)

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    levels = np.linspace(0., 1., 11)
    plt.contourf(xx, yy, Z, cmap=plt.cm.bwr, alpha=0.6, levels=levels)
    cbar = plt.colorbar()
    #cbar.ax.set_ylabel('probability threshold')
    cs1 = plt.contour(xx, yy, Z, colors='k', alpha=0.5, levels=[0.,0.5,1.])
    plt.clabel(cs1, fmt = '%2.1f', colors = 'k', fontsize=14, manual=[(0,1)], inline=1)

    # Plot also the training points
    X_pos = X[y==1]
    X_neg = X[y==0]
    plt.scatter(X_neg[:, 0], X_neg[:, 1], c='b', alpha=0.8, label="Unlabeled")
    plt.scatter(X_pos[:, 0], X_pos[:, 1], c='r', alpha=0.8, label="Positives")

    #plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, alpha=0.8)
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.xlim(-0.2, 1.2)
    plt.ylim(-0.2, 1.2)
    plt.title('{} - Decision boundaries'.format(clf_name))
    plt.legend(loc="upper right")

    #plt.show()
    plt.savefig(os.path.join(out_path, 'boundary__{}'.format(clf_name.replace(' ', ''))), bbox_inches = 'tight')
