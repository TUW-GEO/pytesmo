"""
Provides a temporal matching function
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def df_temp_merge(df_reference, df_other, return_index=False,
                  return_distance=False, tolerance=None, direction='nearest',
                  duplicate_nan=False):
    """
    Merge nearest neighbor between reference and other time series into
    common dataframe.

    Parameters
    ----------
    df_reference : pandas.DataFrame
        Reference time series.
    df_other : tuple/list of pandas.DataFrame or pandas.DataFrame
        Time series matched against reference time series.
    return_index : bool, optional
        Include index of other time series in matched dataframe
        (default: False).
    return_distance : bool, optional
        Include distance information between reference and other time series
        in matched dataframe (default: False).
    tolerance : pd.Timedelta, optional
        Select nearest neighbor tolerance (default: None).
    direction : str, optional
        Whether to search 'backward', 'forward', or 'nearest' matches
        (default: 'nearest').
    duplicate_nan : bool, optional
        Set duplicate NaN (default: False).

    Returns
    -------
    df_tm : pandas.DataFrame
        Reference time series matched with other time series.
    """
    if not isinstance(df_other, tuple):
        df_other = (df_other, )

    if isinstance(df_reference, pd.Series):
        if df_reference.name is None:
            name = 'reference'
        else:
            name = df_reference.name

        df_reference = df_reference.to_frame(name)

    for i, other in enumerate(df_other):

        if isinstance(other, pd.Series):
            if other.name is None:
                name = 'series_{}'.format(i)
            else:
                name = other.name

            other = other.to_frame(name)

        dist_str = 'dist_other_{}'.format(i)
        ind_str = 'ind_other_{}'.format(i)
        other[ind_str] = other.index
        col_other = other.columns

        df = pd.merge_asof(df_reference, other, left_index=True,
                           right_index=True, direction=direction,
                           tolerance=tolerance)

        df[dist_str] = (df[ind_str].values -
                        df.index.values) / np.timedelta64(1, 'D')

        if duplicate_nan:
            unq, unq_idx = np.unique(df[ind_str].values, return_index=True)
            unq_idx = np.concatenate([unq_idx, np.array([len(df)])])
            dist = df[dist_str].values

            no_dup = []
            for j in np.arange(unq_idx.size-1):
                m = np.argmin(np.abs(
                    dist[unq_idx[j]:unq_idx[j+1]])) + unq_idx[j]
                no_dup.append(m)

            duplicates = np.ones(len(df), dtype=np.bool)
            duplicates[no_dup] = False
            df.loc[duplicates, col_other] = np.nan

        fields = []

        if not return_index:
            fields.append(ind_str)

        if not return_distance:
            fields.append(dist_str)

        if fields:
            df.drop(fields, axis=1, inplace=True)

        df_tm = df

    return df_tm


def df_match(reference, *args, **kwds):
    """
    Finds temporal match between the reference pandas.DataFrame (index has to
    be datetime) and n other pandas.DataFrame (index has to be datetime).

    Parameters
    ----------
    reference : pandas.DataFrame or pandas.TimeSeries
        The index of this dataframe will be the reference.
    *args : pandas.DataFrame or pandas.TimeSeries
        The index of this dataframe(s) will be matched.
        If it is a pandas.Series then it has to have a name. Otherwise
        no column name can be assigned to the matched DataFrame.
    window : float
        Fraction of days of the maximum pos./neg. distance allowed, i.e. the
        value of window represents the half-winow size (e.g. window=0.5, will
        search for matches between -12 and +12 hours) (default: None)
    dropna : boolean
        Drop rows containing only NaNs (default: False)
    dropduplicates : boolean
        Drop duplicated temporal matched (default: False)
    asym_window: string, optional
        ``<=`` stands for using a smaller and equal only for the left/smaller
        side of the window comparison
        ``>=`` stands for using a larger and equal only for the right/larger
        side of the window comparison
        The default is to use <= and >= for both sides of the search window

    Returns
    -------
    temporal_matched_args : pandas.DataFrame or tuple of pandas.DataFrame
        Dataframe with index from matched reference index
    """
    if "window" in kwds:
        window = kwds['window']
    else:
        window = None

    if "asym_window" in kwds:
        asym_window = kwds['asym_window']
    else:
        asym_window = None

    temporal_matched_args = []
    ref_step = reference.index.values - reference.index.values[0]

    for arg in args:

        if type(arg) is pd.Series:
            arg = pd.DataFrame(arg)
        comp_step = arg.index.values - reference.index.values[0]

        values = np.arange(comp_step.size)
        # setup kdtree which must get 2D input
        try:
            tree = cKDTree(np.atleast_2d(comp_step).T, balanced_tree=False)
        except TypeError:
            # scipy before version 0.16 does not have the balanced_tree kw
            # but is fast in this case also without it
            tree = cKDTree(np.atleast_2d(comp_step).T)

        dist, i = tree.query(np.atleast_2d(ref_step).T)
        matched = values[i]

        distance = np.zeros_like(matched, dtype=np.float)
        distance.fill(np.nan)
        valid_match = np.invert(np.isnan(matched))

        distance[valid_match] = \
            (arg.index.values[np.int32(matched[valid_match])] -
             reference.index.values[valid_match]) / np.timedelta64(1, 'D')

        arg = arg.assign(index=arg.index.values,
                         merge_key=np.arange(len(arg)))

        arg_matched = pd.DataFrame({'merge_key': matched,
                                    'distance': distance,
                                    'ref_index': reference.index.values})
        arg_matched = arg_matched.merge(arg, on="merge_key", how="left")
        arg_matched.index = arg_matched['ref_index'].values
        arg_matched = arg_matched.sort_index()

        if window is not None:
            if asym_window is None:
                invalid_dist = arg_matched['distance'].abs() > window
            if asym_window == "<=":
                # this means that only distance in the interval [distance[ are
                # taken
                valid_dist = ((arg_matched['distance'] >= 0.0) & (
                    arg_matched['distance'] <= window)) | (
                    (arg_matched['distance'] <= 0.0) & (
                        arg_matched['distance'] > -window))
                invalid_dist = ~valid_dist
            if asym_window == ">=":
                # this means that only distance in the interval ]distance] are
                # taken
                valid_dist = ((arg_matched['distance'] >= 0.0) & (
                    arg_matched['distance'] < window)) | (
                    (arg_matched['distance'] <= 0.0) & (
                        arg_matched['distance'] >= -window))
                invalid_dist = ~valid_dist
            arg_matched.loc[invalid_dist] = np.nan

        if "dropna" in kwds and kwds['dropna']:
            arg_matched = arg_matched.dropna(how='all')

        if "dropduplicates" in kwds and kwds['dropduplicates']:
            arg_matched = arg_matched.dropna(how='all')

            unq, unq_idx = np.unique(arg_matched['index'].values,
                                     return_index=True)
            unq_idx = np.concatenate([unq_idx, np.array([len(arg_matched)])])

            dist = arg_matched['distance'].values
            no_dup = []
            for j in np.arange(unq_idx.size-1):
                m = np.argmin(np.abs(
                    dist[unq_idx[j]:unq_idx[j+1]])) + unq_idx[j]
                no_dup.append(m)

            arg_matched = arg_matched.iloc[no_dup]

        temporal_matched_args.append(
            arg_matched.drop(['merge_key', 'ref_index'], axis=1))

    if len(temporal_matched_args) == 1:
        return temporal_matched_args[0]
    else:
        return tuple(temporal_matched_args)


def matching(reference, *args, **kwargs):
    """
    Finds temporal match between the reference pandas.TimeSeries (index has to
    be datetime) and n other pandas.TimeSeries (index has to be datetime).

    Parameters
    ----------
    reference : pandas.TimeSeries
        The index of this Series will be the reference.
    *args : pandas.TimeSeries
        The index of these Series(s) will be matched.
    window : float
        Fraction of days of the maximum pos./neg. distance allowed, i.e. the
        value of window represents the half-winow size (e.g. window=0.5, will
        search for matches between -12 and +12 hours) (default: None)

    Returns
    -------
    temporal_match : pandas.DataFrame
        containing the index of the reference Series and a column for each
        of the other input Series
    """
    matched_datasets = df_match(reference, *args, dropna=True,
                                dropduplicates=True, **kwargs)

    if type(matched_datasets) != tuple:
        matched_datasets = [matched_datasets]

    matched_data = pd.DataFrame(reference)

    for match in matched_datasets:
        match = match.drop(['distance', 'index'], axis=1)

        if matched_data.index.tz is not None:
            match.index = match.index.tz_localize('utc')

        matched_data = matched_data.join(match)

    return matched_data.dropna()
