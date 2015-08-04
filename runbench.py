from config import config, classifiers
from datasetup import load_data, test_data

# df = load_data(config.data_loc)

# TODO turn df into train_labels, features and test

train_features, test_features, train_labels = test_data()

for clf_name, clf in classifiers.iteritems():
    clf.fit(train_features, train_labels)
    prob_pos = clf.predict_proba(test_features)

    print prob_pos