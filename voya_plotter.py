import logging
from os import system

import pandas
import matplotlib.pyplot as plt
import sklearn.calibration
import sklearn.preprocessing
import sklearn.metrics
import sklearn.decomposition
from sklearn.grid_search import GridSearchCV
import numpy as np
import seaborn
from sklearn.tree import export_graphviz

voya_logger = logging.getLogger('clairvoya')

try:
    import optunity.metrics
    import semisup_metrics
except ImportError:
    voya_logger.info('optunity not installed, disabled roc_pu plot')



class PrInRanking(object):

    def __init__(self, ranking_Frac=1.0, desired_retention = 0.8):
        if ranking_Frac is None:
            self.ranking_Frac = 1.0
        else:
            self.ranking_Frac = ranking_Frac
        if desired_retention is None:
            self.desired_retention = 0.8
        else:
            self.desired_retention = desired_retention
            

    def pr_in_ranking(self, clf, X_test, y_test):

        y_pred = clf.predict_proba(X_test)[:, 1]

        ytuple = pandas.DataFrame(np.column_stack((y_pred, y_test)), columns=['prob', 'label'])
        ytuple = ytuple.sort(columns='prob', ascending=False)

        num_positives_total = np.sum(y_test)
        num_total = y_test.size


        ranking_size = int(num_total*self.ranking_Frac)
        rankedSet = ytuple.iloc[0:ranking_size, :]
        num_positives_inRank = rankedSet[rankedSet.label == 1].shape[0]
        positive_rate = float(num_positives_inRank) / num_positives_total

        return positive_rate
        
    def frac_to_Xpercent(self, clf, X_test, y_test):
        
        for r in np.linspace(0, 1., num=101):
            ranking = PrInRanking(r)
            if ranking.pr_in_ranking(clf, X_test, y_test) > self.desired_retention:
                break;
            print 'A {} fracion of ranking is required to get to desired retention of {}'.format(r, self.desired_retention)
                
        return r





def prVSranking_curve(clf_results):
  
    clf_name = clf_results['clf_name']
    clf = clf_results['clf']
    y_test = clf_results['y_test']
    X_test = clf_results['X_test']
    num_positives_total = np.sum(y_test)
    num_total = y_test.size
  
    pr_curve = np.asarray((0., 0.))  # first point is always (0, 0)
  
    perfect_classifier_pr_curve = np.array([[0., 0.], [float(num_positives_total) / num_total, 1.], [1., 1.]])
    no_classifier_pr_curve = np.array([[0., 0.], [1., 1.]])

    for r in np.linspace(0, 1., num=51):
        ranking = PrInRanking(r)
        pr_curve = np.vstack((pr_curve, np.asarray((r, ranking.pr_in_ranking(clf, X_test, y_test)))))

    # Plot curve
    seaborn.set_style("whitegrid")
    plt.figure(figsize=(7, 7))

    plt.plot(perfect_classifier_pr_curve[:, 0], perfect_classifier_pr_curve[:, 1], label='Perfect Classifier', c='blue')
    plt.plot(no_classifier_pr_curve[:, 0], no_classifier_pr_curve[:, 1], label='No Classifier', c='red')
    plt.plot(pr_curve[:, 0], pr_curve[:, 1], label=clf_name, c='black')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Fraction of Included data (ranked in descending order of probability)')
    plt.ylabel('Fracion of positives found by Classifier')
    plt.title('{} - Positives Found vs Fraction of data'.format(clf_name))
    plt.legend(loc="lower right")
    return pr_curve

def roc_pu(clf_results):
    nboot = 2000
    ci_width = 0.95
    X = clf_results["X_train"]
    y = clf_results["y_train"]
    y_test = clf_results["y_test"]
    clf_name = clf_results["clf_name"]
    clf = clf_results["clf"]
    y_pred = clf_results["y_pred"]

    num_pos = sum(y_test)
    num_neg = 0
    num_unl = len(y_test) - num_pos

    ##############################
    # Estimate "beta", the fraction of positives among the unlabeled.
    # Below we use the method from Elkan & Noto to estimate p(s=1|y=1) = num_pos_labeled/total_pos
    # From here it's simple to determine beta = num_pos_unlabeled/tot_unlabeled
    hold_out_ratio = 0.1  # For now, use default value from E&N method
    positives = np.where(y == 1.)[0]
    hold_out_size = np.ceil(len(positives) * hold_out_ratio)

    if len(positives) <= hold_out_size:
        raise ValueError(
            'Not enough positive examples to estimate p(s=1|y=1,x). Need at least ' + str(hold_out_size + 1) + '.')

    np.random.shuffle(positives)
    hold_out = positives[:hold_out_size]
    X_hold_out = X[hold_out]
    X = np.delete(X, hold_out, 0)
    y = np.delete(y, hold_out)

    clf.fit(X, y)

    hold_out_predictions = clf.predict_proba(X_hold_out)
    hold_out_predictions = hold_out_predictions[:, 1]
    c = np.mean(hold_out_predictions)
    beta = (sum(y_test) / (len(y_test) - sum(y_test))) * (1. - c) / c
    ##############################

    true_pfrac = beta
    num_neg_in_unl = int(round(num_unl * (1 - true_pfrac)))
    num_pos_in_unl = int(round(num_unl * true_pfrac))
    labels = [None] * len(y_test)
    for i in range(len(labels)):
        if (y_test[i] == 1): labels[i] = True
    # labels = [True] * num_pos + [None] * num_pos_in_unl + [False] * num_neg + [None] * num_neg_in_unl
    # true_labels = [True] * (num_pos + num_pos_in_unl) + [False] * (num_neg + num_neg_in_unl)
    # decision_values = generate_pos_class_decvals(num_pos + num_pos_in_unl) + generate_neg_class_decvals(num_neg + num_neg_in_unl)

    num_positives_total = np.sum(y_test)
    num_total = len(y_test)  # y_test.size
    # Computing ROC bounds, curves
    roc_bounds = semisup_metrics.roc_bounds(labels, y_pred,
                                            beta=beta, ci_fun=semisup_metrics.bootstrap_ecdf_bounds)

    fpr, tpr, _ = sklearn.metrics.roc_curve(y_test, y_pred)
    # auc_true, roc_true = optunity.metrics.roc_auc(true_labels, y_pred, return_curve=True)
    auc_neg, curve_neg = optunity.metrics.roc_auc(labels, y_pred, return_curve=True)
    # auc_lower = semisup_metrics.auc(roc_bounds.lower)
    # auc_upper = semisup_metrics.auc(roc_bounds.upper)
    # print('+ Plotting ROC curves.')
    seaborn.set_style("whitegrid")
    plt.figure(figsize=(7, 7))

    # plt.plot(fpr, tpr, color='magenta')
    # plt.plot(*zip(*roc_true), color='black', label="beta = %3.1f" %beta)
    plt.plot(*zip(*roc_bounds.lower), color='blue', label="lower bound")
    plt.plot(*zip(*roc_bounds.upper), color='red', label="upper bound")
    plt.plot(*zip(*curve_neg), color='black', ls=':', label="beta=0")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('%s - PU ROC Bounds with beta=%5.3f (Claesen+2015)' % (clf_name, beta))
    plt.legend(loc="lower right")


def prVSranking_methodComparison(results_dict):

    seaborn.set_style("whitegrid")
    plt.figure(figsize=(7, 7))

    
    for i, (clf_name, clf_results) in enumerate(results_dict.iteritems()):
        y_test = clf_results['y_test']
        #auc_score = clf_results['auc_score']
        #print clf_name, auc_score
        num_positives_total = np.sum(y_test)
        num_total = y_test.size
        pr_curve = prVSranking_curve(clf_results)
        #print pr_curve
        plt.plot(pr_curve[:, 0], pr_curve[:, 1], label=clf_name)

    perfect_classifier_pr_curve = np.array([[0., 0.], [float(num_positives_total) / num_total, 1.], [1., 1.]])
    random_classifier_pr_curve = np.array([[0., 0.], [1., 0.5]])

    plt.plot(perfect_classifier_pr_curve[:, 0], perfect_classifier_pr_curve[:, 1], label='Perfect Classifier', c='blue')
    plt.plot(random_classifier_pr_curve[:, 0], random_classifier_pr_curve[:, 1], label='Random Classifier', c='red')

    plt.xlim([0.0, float(num_positives_total) / num_total + 0.1])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Fraction of Included data (ranked in descending order of probability)')
    plt.ylabel('Fracion of positives found by Classifier')
    plt.title('Positives Found vs Fraction of data')
    plt.legend(loc="lower right")
                
    

def reliability_curve(clf_results):
    """
    :param clf_results: the results dictionary generated by runbench.runbenchamrk

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

    # TODO (ryan) may want to move cm generation code out of here if we also want numeric output
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

    # Display the values of the conf matrix on the plot
    cm_bbox = {'facecolor': 'white', 'alpha': 0.5, 'pad': 10}
    for i in range(2):
        for j in range(2):
            ax1.text(i, j, "%d" % cm[i, j], size=14, ha='center', bbox=cm_bbox)

    plt.colorbar(confMatrix1, ax=ax1)

    ax2.set_title("Normalised Confusion Matrix")
    ax2.set_ylabel('True label')
    ax2.set_xlabel('Predicted label')

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    confMatrix2 = ax2.imshow(cm_normalized, interpolation='nearest', cmap=cmap)

    for i in range(2):
        for j in range(2):
            ax2.text(i, j, "%4.2f" % cm_normalized[i, j], ha='center', size=14, bbox=cm_bbox)

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
    roc_auc = sklearn.metrics.roc_auc_score(y_test, y_pred)

    # Plot of a ROC curve for a specific class
    seaborn.set_style("whitegrid")
    plt.figure(figsize=(7, 7))
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('{} - Receiver operating characteristic'.format(clf_name))
    plt.legend(loc="lower right")


def roc_curve_cv(clf_results, num_folds=6):
    """
    Adapted from http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
    """
    ###############################################################################
    # Run classifier with cross-validation and plot ROC curves

    # TODO (ryan) give filled roc curve bounds instead of loads of lines (option)

    y_train = clf_results["y_train"]
    X_train = clf_results["X_train"]
    clf_name = clf_results["clf_name"]
    clf_notoptimized = clf_results["clf_notoptimized"]
    param_grid = clf_results["param_grid"]

    X = X_train
    y = y_train

    # Do an initial kfold on the data- n_folds will be the number of roc curves shown
    # Can be the same from voya_config
    cv = sklearn.cross_validation.StratifiedKFold(y, n_folds=num_folds)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    seaborn.set_style("whitegrid")
    plt.figure(figsize=(7, 7))

    # Loop over each fold, make distinction on whether grid_search is on/off
    for i, (train, test) in enumerate(cv):
        if param_grid is None:
            clf_fitted = clf_notoptimized.fit(X[train], y[train])

        else:
            # Need a second k-fold here to do Grid Sarch on this particular fold
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

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))  # , label='Luck')

    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = sklearn.metrics.auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

    plt.xlim([0., 1.])
    plt.ylim([0., 1.])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('%s - Receiver operating characteristic CV' % clf_name)
    plt.legend(loc="lower right")


def plot_boundary(clf_results, runPCA=True):
    """ Plots the training data with the first two principle components (or features) with the decision boundries of the
    classifier

    :param runPCA: whether to use the first 2 principle componenets from PCA or the first 2 features in the data frame
    """

    y_train = clf_results["y_train"]
    X_train = clf_results["X_train"]
    clf_name = clf_results["clf_name"]
    clf_notoptimized = clf_results["clf_notoptimized"]

    X = X_train
    y = y_train

    if runPCA:  # then run PCA on the features and take top 2
        X = sklearn.decomposition.PCA(n_components=2).fit_transform(X)
        X = sklearn.preprocessing.MinMaxScaler().fit_transform(X)
        labels = ['pca_1', 'pca_2']
    else:  # select first 2 features
        X = X[:, :2]
        labels = ['x_1', 'x_2']
    # create a mesh to plot in
    h = .01  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    seaborn.set_style("whitegrid")
    plt.figure(figsize=(7, 7))

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    clf_fitted = clf_notoptimized.fit(X, y)

    Z = clf_fitted.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = np.clip(Z, 0., 1.)

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    levels = np.linspace(0., 1., 11)
    plt.contourf(xx, yy, Z, cmap=plt.cm.bwr, alpha=0.6, levels=levels)
    cbar = plt.colorbar()
    # cbar.ax.set_ylabel('probability threshold')
    cs1 = plt.contour(xx, yy, Z, colors='k', alpha=0.5, levels=[0., 0.5, 1.])

    try:
        plt.clabel(cs1, fmt='%2.1f', colors='k', fontsize=14, manual=[(0, 1)], inline=1)
    except UnboundLocalError:
        pass  # in case there is no 0.5 contour in map.

    # Plot also the training points
    X_pos = X[y == 1]
    X_neg = X[y == 0]
    plt.scatter(X_neg[:, 0], X_neg[:, 1], c='b', alpha=0.8, label="Unlabeled")
    plt.scatter(X_pos[:, 0], X_pos[:, 1], c='r', alpha=0.8, label="Positives")

    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.xlim(-0.2, 1.2)
    plt.ylim(-0.2, 1.2)
    plt.title('{} - Decision boundaries'.format(clf_name))
    plt.legend(loc="upper right")


def plot_trees(clf_fitted, feature_names, out_path):
    print("numner of features", clf_fitted.n_features_)
    subdir = "forest-trees"
    mksubdir = "mkdir -p " + out_path + subdir
    system(mksubdir)
    for i, tree in enumerate(clf_fitted.estimators_):
        with open(out_path + subdir + '/RandomForests_tree_' + str(i) + '.dot', 'w') as dotfile:
            export_graphviz(tree, dotfile, feature_names=feature_names, max_depth=4)
            dotfile.close()
            dot2png = "dot -Tpng " + out_path + subdir + "/RandomForests_tree_" + str(
                i) + ".dot -o " + out_path + subdir + "/RandomForests_tree_" + str(i) + ".png"
            system(dot2png)
            rmdot = "rm " + out_path + subdir + "/*.dot"
    system(rmdot)


def pu_search_result(result_file, fig=None):
    """ loads a PU search result file and displays the plot
    :param result_file:
    :return:
    """

    results_df = pandas.read_csv(result_file)

    results_table = results_df.groupby(["clf", "gamma"], as_index=False).agg(['mean', 'std', 'count'])
    colors = seaborn.color_palette("Set2", 10)
    result_classifiers = results_df.clf.unique()

    if fig is None:
        plt.figure(figsize=(10,10))

    for i, clf_name in enumerate(result_classifiers):
        clf_results = results_table.ix[(clf_name)]
        clf_gamma_range = clf_results.index.values
        auc_mean = clf_results.auc["mean"].values
        auc_std = clf_results.auc["std"].values
        auc_count = clf_results.auc["count"].values
        auc_std_err = auc_std / np.sqrt(auc_count)

    #   print '\n---{}\n'.format(clf_name), clf_gamma_range, auc_mean, auc_std_err, colors[i]
        plt.errorbar(clf_gamma_range, auc_mean, label=clf_name,
                     yerr=auc_std_err, c=colors[i], capthick=1)

        plt.scatter(clf_gamma_range, auc_mean, c=colors[i], lw=0)

    plt.ylabel('AUC Score')
    plt.xlabel('Fraction of Unlabelled to Positive')
    plt.legend()
    # plt.xscale('log')
    # title = "Train P {}, N {}".format(
    #     num_pos, num_neg)
    plt.title('PU Search')

    return fig


def pu_search_result_fixed(result_file, err_bars='stderr'):
    """ same as pu_search_result but for a fixed test train (i.e. has 1 row per gamma and the columns std, stderr and folds

    :param err_bars: 'std' or 'std_err'
    :return:
    """

    results_df = pandas.read_csv(result_file)

    results_table = results_df
    colors = seaborn.color_palette("Set2", 10)
    result_classifiers = results_df.clf.unique()

    plt.figure(figsize=(10,10))

    for i, clf_name in enumerate(result_classifiers):
        clf_results = results_table[results_table.clf == clf_name]

        gamma_range = clf_results["gamma"].values
        auc_mean = clf_results["auc"].values
        auc_std = clf_results["std"].values
        auc_std_err = clf_results["stderr"].values
        auc_count = clf_results["folds"].values

        if err_bars == 'std':
            y_err = auc_std
        else:
            y_err = auc_std_err

    #   print '\n---{}\n'.format(clf_name), clf_gamma_range, auc_mean, auc_std_err, colors[i]
        plt.errorbar(gamma_range, auc_mean, label=clf_name, yerr=y_err, c=colors[i], capthick=1)

        plt.scatter(gamma_range, auc_mean, c=colors[i], lw=0)

    plt.ylabel('AUC Score')
    plt.xlabel('Fraction of Unlabelled to Positive')
    plt.legend()
    # plt.xscale('log')
    # title = "Train P {}, N {}".format(
    #     num_pos, num_neg)
    plt.title('PU Search')