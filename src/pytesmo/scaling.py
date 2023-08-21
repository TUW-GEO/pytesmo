"""
Created on Apr 17, 2013

@author: Christoph Paulik christoph.paulik@geo.tuwien.ac.at
"""

from scipy import stats
import numpy as np
import pandas as pd
import scipy.interpolate as sc_int
from warnings import warn
import pytesmo.utils as utils
from pytesmo.cdf_matching import CDFMatching


def add_scaled(df, method="linreg", label_in=None, label_scale=None, **kwargs):
    """
    takes a dataframe and appends a scaled time series to it. If no labels are
    given the first column will be scaled to the second column of the DataFrame

    Parameters
    ----------
    df : pandas.DataFrame
        input dataframe
    method : string
        scaling method
    label_in: string, optional
        the column of the dataframe that should be scaled to that with
        label_scale default is the first column
    label_scale : string, optional
        the column of the dataframe the label_in column should be scaled to
        default is the second column

    Returns
    -------
    df : pandas.DataFrame
        input dataframe with new column labeled label_in+'_scaled_'+method
    """

    if label_in is None:
        label_in = df.columns.values[0]
    if label_scale is None:
        label_scale = df.columns.values[1]

    scaling_func = get_scaling_function(method)

    scaled = scaling_func(
        df[label_in].values, df[label_scale].values, **kwargs
    )

    new_label = label_in + "_scaled_" + method

    df[new_label] = pd.Series(scaled, index=df.index)

    return df


def scale(df, method="linreg", reference_index=0, **kwargs):
    """
    takes pandas.DataFrame and scales all columns to the column specified
    by reference_index with the chosen method

    Parameters
    ----------
    df : pandas.DataFrame
        containing matched time series that should be scaled
    method : string, optional
        method definition, has to be a function in globals() that takes 2
        numpy.array as input and returns one numpy.array of same length
    reference_index : int, optional
        default 0, column index of reference dataset in dataframe

    Returns
    -------
    scaled data : pandas.DataFrame
        all time series of the input DataFrame scaled to the one specified by
        reference_index
    """
    scaling_func = get_scaling_function(method)

    reference = df[df.columns.values[reference_index]]
    df = df.drop([df.columns.values[reference_index]], axis=1)
    # new_df = pd.DataFrame
    for series in df:
        df[series] = pd.Series(
            scaling_func(df[series].values, reference.values, **kwargs),
            index=df.index,
        )

    df.insert(reference_index, reference.name, reference)

    return df


def get_scaling_method_lut():
    """
    Get all defined scaling methods and their function names.

    Returns
    -------
    lut: dictionary
       key: scaling method name
       value: function
    """

    lut = {
        "linreg": linreg,
        "mean_std": mean_std,
        "min_max": min_max,
        "cdf_match": cdf_match,
        "cdf_beta_match": cdf_beta_match,
    }

    return lut


def get_scaling_function(method):
    """
    Get scaling function based on method name.

    Parameters
    ----------
    method: string
        method name as string

    Returns
    -------
    scaling_func: function
        function(src:numpy.ndarray, ref:numpy.ndarray) > scaled_src:np.ndarray

    Raises
    ------
    KeyError:
        if method is not found
    """
    lut = get_scaling_method_lut()
    try:
        return lut[method]
    except KeyError:
        raise KeyError(f"Scaling method {method} not found.")


def min_max(src, ref, **kwargs):
    """
    scales the input datasets so that they have the same minimum
    and maximum afterwards

    Parameters
    ----------
    src : numpy.array
        input dataset which will be scaled
    ref : numpy.array
        src will be scaled to this dataset

    Returns
    -------
    scaled dataset : numpy.array
        dataset src with same maximum and minimum as ref
    """
    return (src - np.min(src)) / (np.max(src) - np.min(src)) * (
        np.max(ref) - np.min(ref)
    ) + np.min(ref)


def linreg_stored_params(src, slope, intercept):
    """
    Scale the input data with passed correction values

    Parameters
    ----------
    src : numpy.array
        Candidate values, that are scaled
    slope : float
        Multiplicative correction value
    intercept : float
        Additive correction value

    Returns
    -------
    src_scaled : numpy.array
        The scaled input values
    """

    return np.abs(slope) * src + intercept


def linreg_params(src, ref):
    """
    Calculate additive and multiplicative correction parameters
    based on linear regression models.

    Parameters
    ----------
    src: numpy.array
        Candidate data (to which the corrections apply)
    ref : numpy.array
        Reference data (which candidate is scaled to)

    Returns
    -------
    slope : float
        Multiplicative correction value
    intercept : float
        Additive correction value
    """

    slope, intercept, r_value, p_value, std_err = stats.linregress(src, ref)

    return slope, intercept


def linreg(src, ref, **kwargs):
    """
    scales the input datasets using linear regression

    Parameters
    ----------
    src : numpy.array
        input dataset which will be scaled
    ref : numpy.array
        src will be scaled to this dataset

    Returns
    -------
    scaled dataset : numpy.array
        dataset scaled using linear regression
    """

    slope, intercept = linreg_params(src, ref)
    return linreg_stored_params(src, slope, intercept)


def mean_std(src, ref, **kwargs):
    """
    scales the input datasets so that they have the same mean
    and standard deviation afterwards

    Parameters
    ----------
    src : numpy.array
        input dataset which will be scaled
    ref : numpy.array
        src will be scaled to this dataset

    Returns
    -------
    scaled dataset : numpy.array
        dataset src with same mean and standard deviation as ref
    """
    return ((src - np.mean(src)) / np.std(src)) * np.std(ref) + np.mean(ref)


@utils.deprecated("Use the new implementation 'cdf_match' instead.")
def cdf_beta_match(*args, **kwargs):
    return cdf_match(*args, **kwargs)


def cdf_match(
        src, ref, nbins=100, minobs=20, linear_edge_scaling=True,
        percentiles=None, combine_invalid=True, max_val=None, min_val=None
):
    """
    Rescales by CDF matching.

    This calculates the empirical CDFs for source and reference dataset using a
    specified number of bins. In case of non-unique percentile values, a beta
    distribution is fitted to the CDF.
    For more robust estimation of the lower and upper bins, linear edge scaling
    is used (see Moesinger et al., 2020 for details).

    Parameters
    ----------
    src: numpy.array
        input dataset which will be scaled
    ref: numpy.array
        src will be scaled to this dataset
    nbins: int, optional
        Number of bins to use for estimation of the CDF
    percentiles : sequence, optional
        Percentile values to use. If this is given, `nbins` is ignored. The
        percentiles might still be changed if `minobs` is given and the number
        data per bin is lower. Default is ``None``.
    minobs : int, optional
        Minimum desired number of observations in a bin for bin resizing. If it
        is ``None`` bins will not be resized. Default is 20.
    linear_edge_scaling : bool, optional
        Whether to derive the edge parameters via linear regression (more
        robust, see Moesinger et al. (2020) for more info). Default is
        ``True``.
        Note that this way only the outliers in the reference (y) CDF are
        handled. Outliers in the input data (x) will not be removed and will
        still show up in the data.
    combine_invalid : bool, optional
        Optional feature to combine the masks of invalid data (NaN, Inf) of
        both source (X) and reference (y) data passed to `fit`. This only makes
        sense if X and y are both timeseries data corresponding to the same
        index. In this case, this makes sures that data is only used if values
        for X and y are available, so that seasonal patterns in missing values
        in one of them do not lead to distortions. (For example, if X is
        available the whole year, but y is only available during summer, the
        distribution of y should not be matched against the whole year CDF of
        X, because that could introduce systematic seasonal biases).
        Default is True.
    max_val, min_val : float, optional
        Maximum and minimum values to enforce.

    Returns
    -------
    CDF matched values: numpy.array
        dataset src with CDF as ref
    """
    matcher = CDFMatching(nbins=nbins, minobs=minobs,
                          linear_edge_scaling=linear_edge_scaling,
                          percentiles=percentiles,
                          combine_invalid=combine_invalid)
    matcher.fit(src, ref)
    scaled = matcher.predict(src)
    if max_val is not None:
        scaled[scaled > max_val] = max_val
    if min_val is not None:
        scaled[scaled < min_val] = min_val
    return scaled
