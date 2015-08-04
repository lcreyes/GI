from sklearn.cross_validation import StratifiedShuffleSplit

###########################################################

def split_train_data(y):
  #X_df = df.values
  #X = X_df[:,4:-1]
  #y = X_df[:,3]

  sss = StratifiedShuffleSplit(y, 10, test_size=0.2, random_state=0)

  return sss

###########################################################
