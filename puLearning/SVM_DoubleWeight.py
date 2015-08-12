
import numpy as np

class SVM_DoubleWeight(object):  
    """
    Runs the second approach described in Elkan & Noto (2008) for training on 
    Positive +  Unlabeled (PU) data, namely:
    
    1. Run a classifier to obtain a probability estimate which is converted 
       into a weight w(x) calculated following eqn 3 from section 3.
    2. For positive-labeled data use weight=1. For unlabeled data consider 
       every element twice: once with weight w(x) and once with weight 1-w(x).
    3. Learn on training data again

    """

    def __init__(self, estimator):
        """
        estimator -- An estimator (SVC) of p(s=1|x):
        """
        self.estimator = estimator
        
        self.unlabeled_weights = []
        
        self.positives = []
        
        self.unlabeled = []
        
        self.estimator_fitted = False
        
        self.weights_available = False
        
        
    def _calculate_weights(self, X, y):
        
        #self.estimator = sklearn.svm.SVC(C=2.5, kernel='linear', 
         #                               class_weight='auto', probability=True)
        
        
        self.estimator.fit(X,y)
        positive_probabilities = self.estimator.predict_proba(X[self.positives])[:,1] 
        unlabeled_probabilities = self.estimator.predict_proba(X[self.unlabeled])[:,1]
        
        # c value from paper = p(s=1|y=1)
        c = np.mean(positive_probabilities)    
        
        self.unlabeled_weights = (1 - c)/c * unlabeled_probabilities /(1.0 - unlabeled_probabilities)
        
        self.weights_available = True
        
        
    def fit (self, X, y):
        
        self.positives = np.where(y==1)[0]
        self.unlabeled = np.where(y==0)[0]
        
        if not self.weights_available:
            self._calculate_weights(X,y)
        
        #define new X set with unlabeled data added twice
        newX = np.vstack((X[self.positives],X[self.unlabeled],X[self.unlabeled]))
        
        
        #define new y set assigning "1" labels for positive data, and for unlabeled 
        #dat: use "1"  for the first set, and then "0" labels for second set
        newy = np.append(y[self.positives], np.ones(y[self.unlabeled].shape))
        newy = np.append(newy, y[self.unlabeled])
                          
                          
        weights = np.append(y[self.positives], self.unlabeled_weights)
        weights = np.append(weights, 1.-self.unlabeled_weights)
        

        self.estimator.fit(newX,newy,sample_weight=weights)
        
        self.estimator_fitted = True
    
    