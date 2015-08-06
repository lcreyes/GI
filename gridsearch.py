

from sklearn.grid_search import GridSearchCV
import voya_config
import numpy as np
import scipy.stats 


lr_parameters = [
  {'C': [1, 10, 100, 1000],'penalty' : ["l1","l2"], 'class_weight' : [None,"auto"]},
 ]

rfc_parameters = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_split": [1, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

svc_parameters = [{'kernel': ['linear'], 'C': [1,10,100]}]




def gridsearch(clf_name,clf_notoptimized) :
    global lr_parameters
    global rfc_parameters
    global svc_parameters

    print("clf_name",clf_name) 
    if (clf_name == "lr"):
        parameters=lr_parameters
    if (clf_name == "rfc"):
        parameters=rfc_parameters
    if (clf_name == "svc"):
        parameters=svc_parameters


    clf_optimized = GridSearchCV(estimator=clf_notoptimized, param_grid=parameters)
    return clf_optimized

