import os.path
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
import seaborn

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
    #print "...done!"
