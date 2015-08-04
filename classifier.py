""" Run classifiers on data
"""
#SVM, LogR, Randomforests


import config as config
import numpy as np
from sklearn import svm

#classifiers = ["SVM", "LogR", "RandomForest"];

#from train_data get features and label, in the meantime random stuff
NumberOfDataPoints = 100
NumberOfTestPoints = 20
NumberOfFeatures = 5

data = np.random.rand(NumberOfDataPoints, NumberOfFeatures)
test = np.random.rand(NumberOfTestPoints, NumberOfFeatures)
label = np.random.randint(2, size=NumberOfDataPoints)
    
def runSVMClassfier(train_data, test_data):
    classifier = svm.NuSVC()
    classifier.fit(data, label)




