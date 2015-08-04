""" Main file to load config and run all classifiers and benchmarks
"""


import pandas as pd


def load_data(fn):
    print(fn)
    df = pd.read_csv(fn)
    return df

# TODO stratifiying

