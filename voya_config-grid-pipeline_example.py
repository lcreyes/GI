""" Quick config file
"""

import sklearn.linear_model
import sklearn.neural_network
import sklearn.pipeline
import sklearn.ensemble
from puLearning.puAdapter import PUAdapter

config = {
    "data_file": "data/test_Gaussians_formatted.csv",
    "out_path": "output",
    "num_folds": 2,
    "test_size": 0.2,
    "auc_folds": 1,
}

# Setup BernoulliRBM Neural net with logistic classifiers
logistic = sklearn.linear_model.LogisticRegression()
#rbm = sklearn.neural_network.BernoulliRBM(verbose=True)
rbm = sklearn.neural_network.BernoulliRBM()
rbm_logistic = sklearn.pipeline.Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])


#estimator = sklearn.svm.SVC(C=10,kernel='rbf',gamma=0.4,probability=True)
#puadaptor = PUAdapter(estimator)
#puestimator = sklearn.pipeline.Pipeline(steps=[('puadaptor',puadaptor)])


pupipeline = sklearn.pipeline.Pipeline([
    ('logistic' , logistic),
    ('puestimator', PUAdapter(logistic)),
])


classifiers = {
    #'Logistic Regression': sklearn.linear_model.LogisticRegression(),
    # 'Random Forests': sklearn.ensemble.RandomForestClassifier(n_estimators=100),
    #'SVC': sklearn.svm.SVC(C=1.0, probability=True),
    # 'rbm_logistic': rbm_logistic,
    # 'Gradient Boosting': sklearn.ensemble.GradientBoostingClassifier(),
    #'puestimator' : PUAdapter(estimator)
    'puestimator' : pupipeline 
}

classifiers_gridparameters = {
    'Logistic Regression': {'C': [0.2, 0.4, 0.6, 0.8, 1.0], 'penalty': ["l1", "l2"], 'class_weight': [None, "auto"]},
    #'Random Forests': {"max_depth": [3, None], "max_features": [1, 3, 10], "min_samples_split": [1, 3, 10],"min_samples_leaf": [1, 3, 10], "bootstrap": [True, False], "criterion": ["gini", "entropy"]},
    # 'SVC': [{'kernel': ['linear'], 'C': [0.2, 0.4, 0.6, 0.8, 1.0]}],
    #'SVC': None,
    # 'rbm_logistic' : {'steps' : [('rbm', rbm), ('logistic', logistic)]},
    #'Gradient Boosting': {"n_estimators": [10, 50, 100], 'learning_rate': [0.1, 0.2, 0.3, 0.5], 'max_depth': [1, 2, 5],}
    #logistic class_weight does not work in pipeline
    'puestimator' : {"logistic__C" : [0.2,0.4,0.6,0.8] ,'logistic__penalty': ["l1", "l2"],"puestimator__hold_out_ratio" : [0.1,0.2,0.3,0.4,0.6,0.8]}
}
