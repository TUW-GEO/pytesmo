# -*- coding: utf-8 -*-
"""
Created on Sep 05 11:00 2019

@author: wolfgang
"""

from pytesmo.metrics import ecol
import pandas as pd
import numpy as np

def create_testdata(n=3):
    np.random.seed = 12345

    df = []
    for i in range(n):
        ds = pd.Series(index=range(100), data=np.random.rand(100))
        ds.name = 'ds{}'.format(i)
        df.append(ds)

    df = pd.concat(df, axis=1)
    return df


def case_ecol():
    df = create_testdata(3)
    results = ecol(df, correlated=[('ds0', 'ds1'), ('ds0', 'ds2')])


if __name__ == '__main__':
    case_ecol()

