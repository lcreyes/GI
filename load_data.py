#!/usr/bin/python

import config
import pandas as pd


def load_data(fn):
    print(fn)
    df = pd.read_csv(fn)
    return df


