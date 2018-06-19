'''
Created on Apr 17, 2013

@author: Christoph Paulik christoph.paulik@geo.tuwien.ac.at
'''

from scipy import stats
import numpy as np
import pandas as pd
import scipy.interpolate as sc_int
from warnings import warn
from pytesmo.utils import unique_percentiles_interpolate


def add_scaled(df, method='linreg', label_in=None, label_scale=None):
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
        the column of the dataframe that should be scaled to that with label_scale
        default is the first column
    label_scale : string, optional
        the column of the dataframe the label_in column should be scaled to
        default is the second column

    Returns
    -------
    df : pandas.DataFrame
        input dataframe with new column labeled label_in+'_scaled_'+method
    """

    if label_in == None:
        label_in = df.columns.values[0]
    if label_scale == None:
        label_scale = df.columns.values[1]

    scaling_func = get_scaling_function(method)

    scaled = scaling_func(df[label_in].values, df[label_scale].values)

    new_label = label_in + '_scaled_' + method

    df[new_label] = pd.Series(scaled, index=df.index)

    return df


def scale(df, method='linreg', reference_index=0):
    """
    takes pandas.DataFrame and scales all columns to the column specified
    by reference_index with the chosen method

    Parameters
    ----------
    df : pandas.DataFrame
        containing matched time series that should be scaled
    method : string, optional
        method definition, has to be a function in globals() that takes 2 numpy.array
        as input and returns one numpy.array of same length
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
    #new_df = pd.DataFrame
    for series in df:
        df[series] = pd.Series(
            scaling_func(df[series].values, reference.values),
            index=df.index)

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

    lut = {'linreg': linreg,
           'mean_std': mean_std,
           'min_max': min_max,
           'lin_cdf_match': lin_cdf_match,
           'cdf_match': cdf_match}
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
    except KeyError as e:
        warn('scaling method not found')
        raise e


def min_max(src, ref):
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
    return ((src - np.min(src)) / (np.max(src) - np.min(src)) *
            (np.max(ref) - np.min(ref)) + np.min(ref))


def linreg_stored_params(src, slope, intercept):
    '''
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
    '''

    return np.abs(slope) * src + intercept


def linreg_params(src, ref):
    '''
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
    '''

    slope, intercept, r_value, p_value, std_err = stats.linregress(src, ref)

    return slope, intercept



def linreg(src, ref):
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


def mean_std(src, ref):
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
    return ((src - np.mean(src)) /
            np.std(src)) * np.std(ref) + np.mean(ref)


def lin_cdf_match(src, ref,
                  min_val=None, max_val=None,
                  percentiles=[0, 5, 10, 30, 50, 70, 90, 95, 100]):
    '''
    computes cumulative density functions of src and ref at their
    respective bin-edges by linear interpolation; then matches CDF of
    src to CDF of ref.

    This function does not make sure that the percentiles are unique so
    it can happen that multiple measurements are scaled to one point or that
    there are NaN values in the output array.

    Parameters
    ----------
    src: numpy.array
        input dataset which will be scaled
    ref: numpy.array
        src will be scaled to this dataset
    min_val: float, optional
        Minimum allowed value, output data is capped at this value
    max_val: float, optional
        Maximum allowed value, output data is capped at this value
    percentiles: list or numpy.ndarray
        Percentiles to use for CDF matching

    Returns
    -------
    CDF matched values: numpy.array
        dataset src with CDF as ref
    '''

    perc_src = np.array(np.percentile(src, percentiles))
    perc_ref = np.array(np.percentile(ref, percentiles))

    return lin_cdf_match_stored_params(src, perc_src, perc_ref,
                                       min_val=min_val, max_val=max_val)


def lin_cdf_match_stored_params(src, perc_src, perc_ref,
                                min_val=None, max_val=None):
    """
    Performs cdf matching using given percentiles.

    Parameters
    ----------
    src: numpy.array
        input data to scale
    perc_src: numpy.array
        percentiles of src estimated through method of choice
    perc_ref: numpy.array
        percentiles of reference data
        estimated through method of choice, must be same size as
        perc_src
    min_val: float, optional
        Minimum allowed value, output data is capped at this value
    max_val: float, optional
        Maximum allowed value, output data is capped at this value
    """

    return gen_cdf_match(src, perc_src, perc_ref,
                         min_val=min_val, max_val=max_val,
                         k=1)


def cdf_match(src, ref,
              min_val=None, max_val=None,
              nbins=100):
    '''
    computes cumulative density functions of src and ref at their
    respective bin-edges by 5th order spline interpolation; then matches CDF of
    src to CDF of ref.

    This function does not make sure that the percentiles are unique so
    it can happen that multiple measurements are scaled to one point or that
    there are NaN values in the output array.

    Parameters
    ----------
    src: numpy.array
        input dataset which will be scaled
    ref: numpy.array
        src will be scaled to this dataset
    min_val: float, optional
        Minimum allowed value, output data is capped at this value
    max_val: float, optional
        Maximum allowed value, output data is capped at this value
    nbins: int, optional
        Number of bins to use for estimation of the CDF

    Returns
    -------
    CDF matched values: numpy.array
        dataset src with CDF as ref
    '''

    percentiles = np.linspace(0, 100, nbins)
    perc_src = np.array(np.percentile(src, percentiles))
    perc_src = unique_percentiles_interpolate(perc_src,
                                              percentiles=percentiles)
    perc_ref = np.array(np.percentile(ref, percentiles))
    perc_ref = unique_percentiles_interpolate(perc_ref,
                                              percentiles=percentiles)

    return gen_cdf_match(src, perc_src, perc_ref,
                         min_val=min_val, max_val=max_val,
                         k=5)


def gen_cdf_match(src,
                  perc_src, perc_ref,
                  min_val=None, max_val=None,
                  k=1):
    """
    General cdf matching:

    1. computes discrete cumulative density functions of
       src- and ref at the given percentiles
    2. computes continuous CDFs by k-th order spline fitting
    3. CDF of src is matched to CDF of ref

    Parameters
    ----------
    src: numpy.array
        input dataset which will be scaled
    perc_src: numpy.array
        percentiles of src
    perc_ref: numpy.array
        percentiles of reference data
        estimated through method of choice, must be same size as
        perc_src
    min_val: float, optional
        Minimum allowed value, output data is capped at this value
    max_val: float, optional
        Maximum allowed value, output data is capped at this value
    k : int, optional
        Order of spline to fit

    Returns
    -------
    CDF matched values: numpy.array
        dataset src with CDF as ref
    """
    # InterpolatedUnivariateSpline uses extrapolation
    # outside of boundaries so all values can be rescaled
    # This is important if the stored percentiles were generated
    # using a subset of the data and the new data has values outside
    # of this original range
    try:
        inter = sc_int.InterpolatedUnivariateSpline(perc_src,
                                                    perc_ref,
                                                    k=k)
    except Exception:
        # here we must catch all exceptions since scipy does not raise a proper
        # Exception
        warn("Too few percentiles for chosen k.")
        return np.full_like(src, np.nan)

    scaled = inter(src)
    if max_val is not None:
        scaled[scaled > max_val] = max_val
    if min_val is not None:
        scaled[scaled < min_val] = min_val

    return scaled
