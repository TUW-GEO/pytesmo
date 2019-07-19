"""
Provides a temporal matching function
"""
import warnings

import numpy as np
import pandas as pd


def df_match(df_reference, df_other, return_index=False,
             return_distance=False, window=None, direction='nearest',
             duplicate_nan=False, dropna=False, dropduplicates=False,
             merge=False):
    """
    Finds temporal match between the reference pandas.DataFrame (index has to
    be datetime) and n other pandas.DataFrame (index has to be datetime).

    Parameters
    ----------
    df_reference : pandas.DataFrame
        Reference time series.
    df_other : tuple/list of pandas.DataFrame or pandas.DataFrame
        Time series matched against reference time series.
    return_index : boolean, optional
        Include index of other time series in matched dataframe
        (default: False).
    return_distance : boolean, optional
        Include distance information between reference and other time series
        in matched dataframe (default: False).
    window : pd.Timedelta or float, optional
        Nearest neighbor tolerance as fraction of days (e.g. 0.5 = 12h) or
        as pandas.Timedelta object (default: None).
    direction : str, optional
        Whether to search
        - 'backward' (-window > t_other - t_ref < 0),
        - 'forward' (0 > t_other - t_ref < window), or
        - 'nearest' (-window > t_other - t_ref < window)
    duplicate_nan : boolean, optional
        Set duplicate NaN (default: False).
    dropna : boolean, optional
        Drop rows containing only NaNs (default: False).
    dropduplicates : boolean, optional
        Drop duplicated temporal matched (default: False).
    merge : boolean, optional
        Merge all other time series columns to reference time series
        (default: False).

    Returns
    -------
    tm_other : pandas.DataFrame or tuple of pandas.DataFrame
        Other time series matched against reference time series.
    """
    if(np.sum(df_reference.index.duplicated()) > 0):
        warnings.warn('Reference time series contains duplicated time stamps,'
                      ' which have been removed.')
        df_reference = df_reference[~df_reference.index.duplicated()]

    if not isinstance(df_other, tuple):
        df_other = (df_other, )

    if isinstance(df_reference, pd.Series):
        if df_reference.name is None:
            name = 'reference'
        else:
            name = df_reference.name

        df_reference = df_reference.to_frame(name)

    if not isinstance(window, pd.Timedelta) and window is not None:
        window = pd.Timedelta(window, unit='D')

    tm_other = []

    for i, other in enumerate(df_other):

        if isinstance(other, pd.Series):
            if other.name is None:
                name = 'series'
            else:
                name = other.name

            other = other.to_frame(name)

        dist_str = 'dist_other'
        ind_str = 'ind_other'

        other[ind_str] = other.index
        col_other = other.columns

        df = pd.merge_asof(df_reference, other, left_index=True,
                           right_index=True, direction=direction,
                           tolerance=window, suffixes=('', '_{}'.format(i)))

        df[dist_str] = (df[ind_str].values -
                        df.index.values) / np.timedelta64(1, 'D')

        if duplicate_nan or dropduplicates:
            unq, unq_idx = np.unique(df[ind_str].dropna().values,
                                     return_index=True)
            unq_idx = np.concatenate([unq_idx, np.array([len(df)])])
            dist = df[dist_str].values

            no_dup = []
            for j in np.arange(unq_idx.size-1):
                m = np.argmin(np.abs(
                    dist[unq_idx[j]:unq_idx[j+1]])) + unq_idx[j]
                no_dup.append(m)

            duplicates = np.ones(len(df), dtype=np.bool)
            duplicates[no_dup] = False

            if duplicate_nan:
                df.loc[duplicates, col_other] = np.nan

            if dropduplicates:
                df = df.loc[~duplicates]

        fields = []

        if not return_index:
            fields.append(ind_str)

        if not return_distance:
            fields.append(dist_str)

        if fields:
            df.drop(fields, axis=1, inplace=True)

        if dropna:
            df.dropna(inplace=True)

        if merge:
            df_reference = df
        else:
            tm_other.append(df)

    if merge:
        tm = df
    else:
        if len(df_other) <= 1:
            tm = df
        else:
            tm = tuple(tm_other)

    return tm


def matching(reference, *args, **kwargs):
    """
    Finds temporal match between the reference time series (index has to
    be datetime) and n other time series (index has to be datetime).

    Parameters
    ----------
    reference : pandas.Series or pandas.DataFrame
        The index of this time series will be the reference.
    *args : pandas.Series or pandas.DataFrame
        The index of these time series will be matched.

    Returns
    -------
    matched_datasets : pandas.DataFrame
        containing the index of the reference time series and a column for
        each of the other input time series.
    """
    matched_datasets = df_match(reference, args, dropna=True,
                                dropduplicates=True, merge=True, **kwargs)

    return matched_datasets.dropna()
