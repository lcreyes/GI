import os.path

import matplotlib.pyplot as plt
import sklearn.calibration
import sklearn.preprocessing
import sklearn.metrics
from sklearn.grid_search import GridSearchCV
import numpy as np
import seaborn

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

    #plt.show()
