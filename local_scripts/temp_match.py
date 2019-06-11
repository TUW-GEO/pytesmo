# -*- coding: utf-8 -*-
from numpy import random
import pandas as pd
import numpy as np

from pytesmo.temporal_matching import matching

COVERAGE_1 = 0.7
COVERAGE_2 = 0.4

def prune(ds, coverage=1.):
    '''
    Randomly remove values from data series to reach the passed coverage

    Parameters:
    --------
    ds : pd.Series
        Pandas series from which we drop values
    coverage : float
        Coverage of ds that is approx. reached after dropping values

    Returns:
    --------
    dat : pd.Series
        The input series with less values.
    '''
    coverage = float(coverage)
    if coverage == 1.:
        return ds

    dat = ds.copy()
    n_drop = dat.index.size - int(dat.index.size * coverage)
    drop_indices = dat.iloc[random.choice(dat.index.size, n_drop)].sort_index().index
    dat.loc[drop_indices] = np.nan
    return dat

    
random.seed(1)
i1 = pd.date_range(start='2000-01-01T00:00', end='2000-06-30T00:00', freq='D')
s1 = pd.Series(index=i1, data=random.rand(i1.size))
d1 = prune(ds=s1, coverage=COVERAGE_1)

i2 = pd.date_range(start='2000-01-01T00:00', end='2000-06-30T00:00', freq='3H')
s2 = pd.Series(index=i2, data=random.rand(i2.size))
d2 = prune(ds=s2, coverage=COVERAGE_2)


matched = matching(d2, (d1))
