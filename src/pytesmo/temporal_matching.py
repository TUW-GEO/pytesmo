"""
Provides a temporal matching function
"""

from itertools import product
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
        ``<=`` stands for using a smaller and equal only for the left/smaller
        side of the window comparison, ``>=`` stands for using a larger and
        equal only for the right/larger side of the window comparison.
        The default is to use <= and >= for both sides of the search window

    Returns
    -------
    temporal_matched_args : pandas.DataFrame or tuple of pandas.DataFrame
        Dataframe with index from matched reference index
    """
    warnings.warn(
        "'pytesmo.temporal_matching.df_match' is deprecated. Use"
        "'pytesmo.temporal_matching.temporal_collocation' instead!",
        DeprecationWarning,
    )

    if "window" in kwds:
        window = kwds["window"]
    else:
        window = None

    if "asym_window" in kwds:
        asym_window = kwds["asym_window"]
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

        distance = np.zeros_like(matched, dtype=float)
        distance.fill(np.nan)
        valid_match = np.invert(np.isnan(matched))

        distance[valid_match] = (
            arg.index.values[np.int32(matched[valid_match])]
            - reference.index.values[valid_match]
        ) / np.timedelta64(1, "D")

        arg = arg.assign(index=arg.index.values, merge_key=np.arange(len(arg)))

        arg_matched = pd.DataFrame(
            {
                "merge_key": matched,
                "distance": distance,
                "ref_index": reference.index.values,
            }
        )
        arg_matched = arg_matched.merge(arg, on="merge_key", how="left")
        arg_matched.index = arg_matched["ref_index"].values
        arg_matched = arg_matched.sort_index()

        if window is not None:
            if asym_window is None:
                invalid_dist = arg_matched["distance"].abs() > window
            if asym_window == "<=":
                # this means that only distance in the interval [distance[ are
                # taken
                valid_dist = (
                    (arg_matched["distance"] >= 0.0)
                    & (arg_matched["distance"] <= window)
                ) | (
                    (arg_matched["distance"] <= 0.0)
                    & (arg_matched["distance"] > -window)
                )
                invalid_dist = ~valid_dist
            if asym_window == ">=":
                # this means that only distance in the interval ]distance] are
                # taken
                valid_dist = (
                    (arg_matched["distance"] >= 0.0)
                    & (arg_matched["distance"] < window)
                ) | (
                    (arg_matched["distance"] <= 0.0)
                    & (arg_matched["distance"] >= -window)
                )
                invalid_dist = ~valid_dist
            arg_matched.loc[invalid_dist] = np.nan

        if "dropna" in kwds and kwds["dropna"]:
            arg_matched = arg_matched.dropna(how="all")

        if "dropduplicates" in kwds and kwds["dropduplicates"]:
            arg_matched = arg_matched.dropna(how="all")
            g = arg_matched.groupby("merge_key")
            min_dists = g.distance.apply(lambda x: x.abs().idxmin())
            arg_matched = arg_matched.loc[min_dists]

        temporal_matched_args.append(
            arg_matched.drop(["merge_key", "ref_index"], axis=1)
        )

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
        containing the index of the reference Series and a column for each of
        the other input Series
    """
    warnings.warn(
        "'pytesmo.temporal_matching.matching' is deprecated. Use"
        "'pytesmo.temporal_matching.temporal_collocation' instead!",
        DeprecationWarning,
    )
    matched_datasets = df_match(
        reference, *args, dropna=True, dropduplicates=True, **kwargs
    )

    if type(matched_datasets) != tuple:
        matched_datasets = [matched_datasets]

    matched_data = pd.DataFrame(reference)

    for match in matched_datasets:
        match = match.drop(["distance", "index"], axis=1)
        matched_data = matched_data.join(match)

    return matched_data.dropna()


def temporal_collocation(
    reference,
    other,
    window,
    method="nearest",
    return_index=False,
    return_distance=False,
    dropduplicates=False,
    dropna=False,
    checkna=False,
    flag=None,
    use_invalid=False,
):
    """
    Temporally collocates values to reference.

    Parameters
    ----------
    reference : pd.DataFrame, pd.Series, or pd.DatetimeIndex
        The reference onto which `other` should be collocated. If this is a
        DataFrame or a Series, the index must be a DatetimeIndex. If the index
        is timezone-naive and `other` is not, the timezone of `other` will be
        assumed.
    other : pd.DataFrame or pd.Series
        Data to be collocated. Must have a pd.DatetimeIndex as index. If the
        index is timezone-naive and `reference` is not, the timezone of the
        reference data will be assumed.
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
        This uses ``how="all"``, that is, only rows where all values are NaN
        are dropped. Default is ``False``.
    checkna: bool, optional
        Whether to check if only NaNs are returned (i.e. no match has been
        found). If set to ``True``, raises a ``UserWarning`` in case no match
        has been found. Default is ``False``.
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
        raise ValueError("'other' must be pd.DataFrame or pd.Series.")
    if not isinstance(window, pd.Timedelta):
        window = pd.Timedelta(days=window)
    if flag is not None:
        if isinstance(flag, str):
            flag = other[flag].values
        if len(flag) != len(ref_dr):  # pragma: no cover
            raise ValueError("Flag must have same length as reference")
        flagged = flag.astype(bool)
        has_invalid = np.any(flagged)
    else:
        has_invalid = False

    # preprocessing
    # ------------
    if ref_dr.tz is None and other.index.tz is None:
        # no timezone info provided for any of the inputs, so we will continue
        # to use timezone naive frames
        pass
    else:
        if ref_dr.tz is None:
            ref_dr = ref_dr.tz_localize(other.index.tz)
            warnings.warn(
                "No timezone given for reference, assuming it's in the same"
                f" timezone as other, {other.index.tz}.",
                UserWarning,
            )
        elif other.index.tz is None:
            other = other.tz_localize(ref_dr.tz)
            warnings.warn(
                "No timezone given for other, assuming it's in the same"
                f" timezone as reference, {other.index.tz}.",
                UserWarning,
            )
        if other.index.tz != ref_dr.tz:
            other = other.tz_convert(ref_dr.tz)
    if dropduplicates or method == "nearest":
        other = other[~other.index.duplicated(keep="first")]
        ref_duplicated = ref_dr.duplicated(keep="first")
        if np.any(ref_duplicated):
            warnings.warn(
                "Dropping duplicated indices in reference."
                " This might indicate issues with your data."
            )
            ref_dr = ref_dr[~ref_dr.duplicated(keep="first")]

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
    if checkna:
        if np.any(collocated.isnull().apply(np.all)):
            warnings.warn("No match has been found")
    if dropna:
        collocated.dropna(inplace=True, how="all")

    return collocated


def combined_temporal_collocation(
    reference,
    others,
    window,
    method="nearest",
    dropduplicates=False,
    dropna=False,
    combined_dropna=False,
    flag=None,
    checkna=False,
    use_invalid=False,
    add_ref_data=False,
):
    """
    Temporally collocates multiple dataframes to reference times.

    Parameters
    ----------
    reference : pd.DataFrame, pd.Series, or pd.DatetimeIndex
        The reference onto which `other` should be collocated. If this is a
        DataFrame or a Series, the index must be a DatetimeIndex. If the index
        is timezone-naive, UTC will be assumed.
    others : list/tuple of pd.DataFrame or pd.Series
        DataFrames/Series to be collocated. Each entry must have a
        pd.DatetimeIndex as index. If the index is timezone-naive, the timezone
        of the reference data will be assumed.
    window : pd.Timedelta or float
        Window around reference timestamps in which to look for data. Floats
        are interpreted as number of days.
    method : str, optional
        Which method to use for the temporal collocation:

        - "nearest" (default): Uses the nearest valid neighbour. When this
          method is used, entries with duplicate index values in `other` will
          be dropped, and only the first of the duplicates is kept.

    dropduplicates : bool, optional
        Whether to drop duplicated timestamps in `others`. Default is
        ``False``, except when ``method="nearest"``, in which case this is
        enforced to be ``True``.
    dropna : bool, optional
        Whether to drop NaNs from the resulting dataframe (arising for example
        from duplicates with ``duplicates_nan=True`` or from missing values).
        Default is ``False``.
    combined_dropna : str or bool, optional
        Whether and how to drop NaNs from the resulting combined DataFrame. Can
        be ``"any"``, ``"all"``, ``True`` or ``False``. "any" makes sure that
        the output dataframe only has values at times where all input frames
        had values, while "all" only drops lines where all values are NaN.
        ``True`` is the same as "any", and ``False`` (default) disables
        dropping NaNs.
    checkna: bool, optional
        Whether to check if only NaNs are returned (i.e. no match has been
        found). If set to ``True``, raises a ``UserWarning`` in case no match
        has been found. Default is ``False``.
    flag : np.ndarray or None, optional
        Flag column as array. If this is given, the column will be interpreted
        as validity indicator. Any nonzero values mark the row as invalid.
        Default is ``None``.
    use_invalid : bool, optional
        Whether to use invalid values marked by `flag` in case no valid values
        are available. Default is ``False``.
    add_ref_data : bool, optional
        If `reference` is a DataFrame or Series, add the data to the final
        collocated dataframe.

    Returns
    -------
    collocated : pd.DataFrame or pd.Series
        Temporally collocated DataFrame with variables from all input frames
        merged together.
    """
    dfs = [
        temporal_collocation(
            reference,
            other,
            window,
            method=method,
            return_index=False,
            return_distance=False,
            dropduplicates=dropduplicates,
            dropna=dropna,
            checkna=checkna,
            flag=flag,
            use_invalid=use_invalid,
        )
        for other in others
    ]
    if isinstance(reference, (pd.DataFrame, pd.Series)) and add_ref_data:
        # first, check if we have to remove duplicates
        if dropduplicates:
            duplicated = reference.index.duplicated(keep="first")
            reference = reference[~duplicated]
        dfs.insert(0, reference)

    # Before merging we have to check if the timezones are consistent.
    timezones = [d.index.tz for d in dfs]
    uniq_tzs = set(timezones)
    if len(uniq_tzs) == 1:
        pass
    else:
        actual_tzs = set(tz for tz in timezones if tz is not None)
        if len(actual_tzs) == 1:
            # convert all to common tz
            tz = list(actual_tzs)[0]
        else:
            # Multiple different timezones, convert all to UTC and raise a
            # warning
            tz = "UTC"
            warnings.warn(
                "Input DataFrames have mixed timezones, converting everything"
                " to UTC.",
                UserWarning,
            )
        for d in dfs:
            if d.index.tz is None:
                d.index = d.index.tz_localize(tz)
            else:
                d.index = d.index.tz_convert(tz)

    merged = pd.concat(dfs, axis=1)
    if combined_dropna:
        if combined_dropna is True:
            combined_dropna = "any"
        merged = merged.dropna(how=combined_dropna)
    return merged


def dfdict_combined_temporal_collocation(
    dfs, refname, window=None, n=None, **kwargs
):
    """
    Applies :py:func:`combined_temporal_collocation` on a dictionary of
    dataframes.

    Parameters
    ----------
    dfs : dict
        Dictionary of pd.DataFrames containing the dataframes to be collocated.
    refname : str
        Name of the reference frame in `dfs`.
    window : pd.Timedelta or float, optional
        Window around reference timestamps in which to look for data. Floats
        are interpreted as number of days. If it is not given, defaults to 1
        hour to mimick the behaviour of
        ``BasicTemporalMatching.combinatory_matcher``.
    n : dummy argument
        Will be ignored.
    **kwargs :
        Keyword arguments passed to :py:func:`combined_temporal_collocation`.

    Returns:
    --------
    matched_dict : dict
        Dictionary where the keys are tuples of ``(other_name, refname)`` for
        each other key in `dfs`, and values are the matched dataframes.
        The column names of the dataframes are again tuples of ``(name, col)``
        where `name` is the key from `dfs` and `col` is the original column
        name in the input dataframe.
    """
    if window is None:
        window = pd.Timedelta(hours=1)

    others = []
    for name in dfs:
        if name != refname:
            others.append(df_name_multiindex(dfs[name], name))
    ref = df_name_multiindex(dfs[refname], refname)
    matched_df = combined_temporal_collocation(
        ref, others, window, add_ref_data=True, combined_dropna=True, **kwargs
    )

    # unpack again to dictionary
    matched_dict = {}
    for name in dfs:
        if name != refname:
            tuple_keys = list(product((name,), dfs[name].columns))
            keep = list(ref.columns) + tuple_keys
            matched_dict[(name, refname)] = matched_df[keep]
            # rename_dict = {tk: tk[1] for tk in tuple_keys}
            # matched_dict[(refname, name)].rename(rename_dict, axis=1,
            # inplace=True)
    return matched_dict


def df_name_multiindex(df, name):
    """
    Rename columns of a DataFrame by using new column names that
    are tuples of (name, column_name) to ensure unique column names
    that can also be split again. This transforms the columns to a MultiIndex.
    """
    d = {}
    for c in df.columns:
        d[c] = (name, c)

    return df.rename(columns=d)
