"""
Provides a temporal matching function
"""

import numpy as np
import pandas as pd
from pykdtree.kdtree import KDTree
import warnings


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
        ``<=`` stands for using a smaller and equal only for the left/smaller side of the window comparison
        ``>=`` stands for using a larger and equal only for the right/larger side of the window comparison
        The default is to use <= and >= for both sides of the search window

    Returns
    -------
    temporal_matched_args : pandas.DataFrame or tuple of pandas.DataFrame
        Dataframe with index from matched reference index
    """
    warnings.warn(
        "'pytesmo.temporal_matching.df_match' is deprecated. Use"
        "'pytesmo.temporal_matching.temporal_collocation' instead!",
        DeprecationWarning
    )

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
            tree = KDTree(np.atleast_2d(comp_step).T, balanced_tree=False)
        except TypeError:
            # scipy before version 0.16 does not have the balanced_tree kw
            # but is fast in this case also without it
            tree = KDTree(np.atleast_2d(comp_step).T)

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
                valid_dist = ((arg_matched['distance'] >= 0.0) & (arg_matched['distance'] <= window)) | (
                    (arg_matched['distance'] <= 0.0) & (arg_matched['distance'] > -window))
                invalid_dist = ~valid_dist
            if asym_window == ">=":
                # this means that only distance in the interval ]distance] are
                # taken
                valid_dist = ((arg_matched['distance'] >= 0.0) & (arg_matched['distance'] < window)) | (
                    (arg_matched['distance'] <= 0.0) & (arg_matched['distance'] >= -window))
                invalid_dist = ~valid_dist
            arg_matched.loc[invalid_dist] = np.nan

        if "dropna" in kwds and kwds['dropna']:
            arg_matched = arg_matched.dropna(how='all')

        if "dropduplicates" in kwds and kwds['dropduplicates']:
            arg_matched = arg_matched.dropna(how='all')
            g = arg_matched.groupby('merge_key')
            min_dists = g.distance.apply(lambda x: x.abs().idxmin())
            arg_matched = arg_matched.loc[min_dists]

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
        containing the index of the reference Series and a column for each of the
        other input Series
    """
    warnings.warn(
        "'pytesmo.temporal_matching.matching' is deprecated. Use"
        "'pytesmo.temporal_matching.temporal_collocation' instead!",
        DeprecationWarning
    )
    matched_datasets = df_match(reference, *args, dropna=True,
                                dropduplicates=True, **kwargs)

    if type(matched_datasets) != tuple:
        matched_datasets = [matched_datasets]

    matched_data = pd.DataFrame(reference)

    for match in matched_datasets:
        match = match.drop(['distance', 'index'], axis=1)
        matched_data = matched_data.join(match)

    return matched_data.dropna()


def temporal_collocation(reference, other, window, method="nearest",
                         return_index=False, return_distance=False,
                         dropduplicates=False, dropna=False, flag=None,
                         use_invalid=False):
    """
    Temporally collocates values to reference.

    Parameters
    ----------
    reference : pd.DataFrame, pd.Series, or pd.DatetimeIndex
        The reference onto which `other` should be collocated. If this is a
        DataFrame or a Series, the index must be a DatetimeIndex. If the index
        is timezone-naive, UTC will be assumed.
    other : pd.DataFrame or pd.Series
        Data to be collocated. Must have a pd.DatetimeIndex as index. If the
        index is timezone-naive, the timezone of the reference data will be
        assumed.
    window : pd.Timedelta or float
        Window around reference timestamps in which to look for data. Floats
        are interpreted as number of days.
    method : str, optional
        Which method to use for the temporal collocation:

        - "nearest" (default): Uses the nearest valid neighbour. When this
          method is used, entries with duplicate index values in `other` will
          be dropped, and only the first of the duplicates is kept.

    return_index : boolean, optional
        Include index of `other` in matched dataframe (default: False). Only
        used with ``method="nearest"``. The index will be added as a separate
        column with the name "index_other".
    return_distance : boolean, optional
        Include distance information between `reference` and `other` in matched
        dataframe (default: False). This is only used with
        ``method="nearest"``, and implies ``return_index=True``. The distance
        will be added as a separate column with the name "distance_other".
    dropduplicates : bool, optional
        Whether to drop duplicated timestamps in `other`. Default is ``False``,
        except when ``method="nearest"``, in which case this is enforced to be
        ``True``.
    dropna : bool, optional
        Whether to drop NaNs from the resulting dataframe (arising for example
        from duplicates with ``duplicates_nan=True`` or from missing values).
        Default is ``False``
    flag : np.ndarray, str or None, optional
        Flag column as array or name of the flag column in `other`. If this is
        given, the column will be interpreted as validity indicator. Any
        nonzero values mark the row as invalid. Default is ``None``.
    use_invalid : bool, optional
        Whether to use invalid values marked by `flag` in case no valid values
        are available. Default is ``False``.

    Returns
    -------
    collocated : pd.DataFrame or pd.Series
        Temporally collocated version of ``other``.
    """

    # input validation
    # ----------------
    if isinstance(reference, (pd.Series, pd.DataFrame)):
        ref_dr = reference.index
    elif isinstance(reference, pd.DatetimeIndex):
        ref_dr = reference
    else:  # pragma: no cover
        raise ValueError(
            "'reference' must be pd.DataFrame, pd.Series, or pd.DatetimeIndex."
        )
    if not isinstance(other, (pd.Series, pd.DataFrame)):  # pragma: no cover
        raise ValueError(
            "'other' must be pd.DataFrame or pd.Series."
        )
    if not isinstance(window, pd.Timedelta):
        window = pd.Timedelta(days=window)
    if flag is not None:
        if isinstance(flag, str):
            flag = other[flag].values
        if len(flag) != len(ref_dr):  # pragma: no cover
            raise ValueError(
                "Flag must have same length as reference"
            )
        flagged = flag.astype(np.bool)
        has_invalid = np.any(flagged)
    else:
        has_invalid = False


    # preprocessing
    # ------------
    if ref_dr.tz is None:
        ref_dr = ref_dr.tz_localize("UTC")
    if other.index.tz is None:
        other = other.tz_localize(ref_dr.tz)
    if other.index.tz != ref_dr.tz:
        other = other.tz_convert(ref_dr.tz)
    if dropduplicates or method == "nearest":
        other = other[~other.index.duplicated(keep="first")]

    # collocation
    # -----------
    if method == "nearest":
        # Nearest neighbour collocation, uses pandas reindex

        if return_index or return_distance:
            new_cols = {}
            new_cols["index_other"] = other.index
            if return_distance:
                new_cols["distance_other"] = np.zeros(len(other))
            other = other.assign(**new_cols)

        def collocate(df):
            return df.reindex(ref_dr, method="nearest", tolerance=window)

        if has_invalid:
            collocated = collocate(other[~flagged])
            if use_invalid:
                invalid = collocate(other[flagged])
                collocated = collocated.combine_first(invalid)
        else:
            collocated = collocate(other)

        if return_distance:
            collocated["distance_other"] = (
                collocated["index_other"] - collocated.index
            )

    else:
        raise NotImplementedError(
            "Only nearest neighbour collocation is implemented so far"
        )

    # postprocessing
    # --------------
    if dropna:
        collocated.dropna(inplace=True)

    return collocated
