# Copyright (c) 2015,Vienna University of Technology,
# Department of Geodesy and Geoinformation
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of the Vienna University of Technology,
#      Department of Geodesy and Geoinformation nor the
#      names of its contributors may be used to endorse or promote products
#      derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL VIENNA UNIVERSITY OF TECHNOLOGY,
# DEPARTMENT OF GEODESY AND GEOINFORMATION BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
Module containing utility functions that do not fit into other modules
'''
import numpy as np
import scipy.interpolate as sc_int
import scipy.optimize as sc_opt
import scipy.special as sc_special

import functools
import inspect
import warnings
import os
from pathlib import Path


def rootdir() -> Path:
    return Path(os.path.join(os.path.dirname(
        os.path.abspath(__file__)))).parents[1]


def deprecated(message: str = None):
    """
    Decorator for classes or functions to mark them as deprecated.
    If the decorator is applied without a specific message (`@deprecated()`),
    the default warning is shown when using the function/class. To specify
    a custom message use it like:
        @deprecated('Don't use this function anymore!').

    Parameters
    ----------
    message : str, optional (default: None)
        Custom message to show with the DeprecationWarning.
    """

    def decorator(src):
        default_msg = f"Pytesmo " \
                      f"{'class' if inspect.isclass(src) else 'method'} " \
                      f"'{src.__module__}.{src.__name__}' " \
                      f"is deprecated and will be removed soon."

        @functools.wraps(src)
        def new_func(*args, **kwargs):
            warnings.simplefilter('always', DeprecationWarning)

            warnings.warn(
                default_msg if message is None else message,
                category=DeprecationWarning,
                stacklevel=2)
            warnings.simplefilter('default', DeprecationWarning)
            return src(*args, **kwargs)

        return new_func

    return decorator


def ml_percentile(in_data, percentiles):
    """
    Calculate percentiles in the way Matlab and IDL do it.

    By using interpolation between the lowest an highest rank and the
    minimum and maximum outside.

    Parameters
    ----------
    in_data: numpy.ndarray
        input data
    percentiles: numpy.ndarray
        percentiles at which to calculate the values

    Returns
    -------
    perc: numpy.ndarray
        values of the percentiles
    """

    data = np.sort(in_data)
    p_rank = 100.0 * (np.arange(data.size) + 0.5) / data.size
    perc = np.interp(percentiles, p_rank, data, left=data[0], right=data[-1])
    return perc


def interp_uniq(src):
    """
    replace non unique values by their linear interpolated value
    This method interpolates iteratively like it is done in IDL.

    Parameters
    ----------
    src: numpy.array
        array to ensure uniqueness of

    Returns
    -------
    src: numpy.array
        interpolated unique values in array of same size as src

    """
    size = len(src)
    uniq, uniq_ind, counts = np.unique(
        src, return_index=True, return_counts=True)

    while len(src[uniq_ind]) != size:
        # replace non unique percentiles by their linear interpolated value
        # This method interpolates iteratively like it is done in IDL
        # and might be replaced by a faster method of simple linear
        # interpolation
        for i in range(len(uniq_ind)):
            pos = np.where(src == src[uniq_ind[i]])[0]
            if len(pos) > 1:
                if pos[0] == 0 and pos[-1] < size - 1:
                    src[pos[-1]] = (src[pos[len(pos) - 2]] +
                                    src[pos[-1] + 1]) / 2.0
                elif pos[-1] == size - 1:
                    src[pos[0]] = (src[pos[1]] + src[pos[0] - 1]) / 2.0
                else:
                    src[pos[0]] = (src[pos[1]] + src[pos[0] - 1]) / 2.0
                    src[pos[1]] = (src[pos[0]] + src[pos[1] + 1]) / 2.0
            uniq_ind = np.unique(src, return_index=True)[1]

    return src


def unique_percentiles_interpolate(perc_values,
                                   percentiles=[
                                       0, 5, 10, 30, 50, 70, 90, 95, 100
                                   ],
                                   k=1):
    """
    Try to ensure that percentile values are unique
    and have values for the given percentiles.

    If only all the values in perc_values are the same.
    The array is unchanged.

    Parameters
    ----------
    perc_values: list or numpy.ndarray
        calculated values for the given percentiles
    percentiles: list or numpy.ndarray
        Percentiles to use for CDF matching
    k: int
        Degree of spline interpolation to use for
        filling duplicate percentile values

    Returns
    -------
    uniq_perc_values: numpy.ndarray
        Unique percentile values generated through linear
        interpolation over removed duplicate percentile values
    """
    uniq_ind = np.sort(np.unique(perc_values, return_index=True)[1])
    if len(uniq_ind) == 1:
        uniq_ind = np.repeat(uniq_ind, 2)
    uniq_ind[-1] = len(percentiles) - 1
    uniq_perc_values = perc_values[uniq_ind]
    inter = sc_int.InterpolatedUnivariateSpline(
        np.array(percentiles)[uniq_ind],
        uniq_perc_values,
        k=k,
        ext=0,
        check_finite=True)
    uniq_perc_values = inter(percentiles)
    return uniq_perc_values


def unique_percentiles_beta(perc_values, percentiles):
    """
    Compute unique percentile values
    by fitting the CDF of a beta distribution to the
    percentiles.

    Parameters
    ----------
    perc_values: list or numpy.ndarray
        calculated values for the given percentiles
    percentiles: list or numpy.ndarray
        Percentiles to use for CDF matching

    Returns
    -------
    uniq_perc_values: numpy.ndarray
        Unique percentile values generated through fitting
        the CDF of a beta distribution.

    Raises
    ------
    RuntimeError
        If no fit could be found.
    """
    # normalize between 0 and 1
    uniq, uniq_ind, counts = np.unique(
        perc_values, return_index=True, return_counts=True)
    if len(uniq) != len(perc_values):
        min_value = np.min(perc_values)
        perc_values = perc_values - min_value
        max_value = np.max(perc_values)
        perc_values = perc_values / max_value
        percentiles = np.asanyarray(percentiles)
        percentiles = percentiles / 100.0

        p, ier = sc_opt.curve_fit(betainc, percentiles, perc_values)
        uniq_perc_values = sc_special.betainc(p[0], p[1], percentiles)
        uniq_perc_values = uniq_perc_values * max_value + min_value
    else:
        uniq_perc_values = perc_values
    return uniq_perc_values


def betainc(x, a, b):
    return sc_special.betainc(a, b, x)


def element_iterable(el):
    """
    Test if a element is iterable

    Parameters
    ----------
    el: object


    Returns
    -------
    iterable: boolean
       if True then then el is iterable
       if Fales then not
    """
    try:
        el[0]
        iterable = True
    except (TypeError, IndexError):
        iterable = False

    return iterable


def ensure_iterable(el):
    """
    Ensure that an object is iterable by putting it into a list.
    Strings are handled slightly differently. They are
    technically iterable but we want to keep the whole.

    Parameters
    ----------
    el: object

    Returns
    -------
    iterable: list
        [el]
    """
    if type(el) == str:
        return [el]
    if not element_iterable(el):
        return [el]
    else:
        return el


def array_dropna(*arrs):
    """
    Drop elements from input arrays where ANY array is NaN

    Parameters
    ----------
    *arrs : np.array(s)
        One or multiple numpy arrays of the same length that contain nans

    Returns
    -------
    arrs_dropna : np.array
        Input arrays without NaNs
    """

    idx = ~np.logical_or(*[np.isnan(x) for x in arrs])
    arrs_dropna = [np.compress(idx, x) for x in arrs]

    if len(arrs_dropna) == 1:
        arrs_dropna = arrs_dropna[0]

    return tuple(arrs_dropna)


def derive_edge_parameters(src, ref, perc_src, perc_ref):
    '''
    Method to compute the regression parameters and new percentile values for
    the edge matching in CDF matching, based on a linear scaling model.

    Parameters
    ----------
    src : numpy.array
        input dataset which will be scaled
    ref : numpy.array
        src will be scaled to this dataset
    edge_src : list
        list with low and high edges (in percentile values) of src
    edge_ref : list
        list with low and high edges (in percentile values) of ref

    Returns
    -------
    parms_lo : tuple
        slope and intercept parameters to scale the lower edge.
    parms_hi : tuple
        slope and intercept parameters to scale the higher edge.
    perc_ref : list-like
        new percentile values after regression
    '''
    # select higher and lower edges
    x_lo = src[src <= perc_src[1]] - perc_src[1]
    y_lo = ref[ref <= perc_ref[1]] - perc_ref[1]
    x_hi = src[src >= perc_src[-2]] - perc_src[-2]
    y_hi = ref[ref >= perc_ref[-2]] - perc_ref[-2]

    # calculate least squares regression parameters
    def return_regress(x, y, where, perc_src=perc_src, perc_ref=perc_ref):
        n = min(len(x), len(y))
        x, y = x[:n], y[:n]
        x, y = np.sort(x), np.sort(y)
        slope, res, rank, s = np.linalg.lstsq(x.reshape(-1, 1), y, rcond=None)
        if where == 'low':
            intercept = perc_ref[1] - slope[0] * perc_src[1]
        elif where == 'high':
            intercept = perc_ref[-2] - slope[0] * perc_src[-2]

        return slope[0], intercept

    parms_lo = return_regress(x_lo, y_lo, 'low')
    parms_hi = return_regress(x_hi, y_hi, 'high')

    perc_ref[0] = perc_ref[1] + parms_lo[0] * (perc_src[0] - perc_src[1])
    perc_ref[-1] = perc_ref[-2] + parms_hi[0] * (perc_src[-1] - perc_src[-2])

    return parms_lo, parms_hi, perc_ref


def scale_edges(scaled, src, ref, perc_src, perc_ref):
    '''
    Method to scale the edges of the src timeseries using a linear regression
    method based on Moesinger et al. (2020).

    Parameters
    ----------
    scaled : numpy.array
        scaled array where edge values should be replaced
    src : numpy.array
        input dataset which will be scaled.
    ref : numpy.array
        src will be scaled to this dataset
    perc_src : numpy.array
        percentiles of src
    perc_ref : numpy.array
        percentiles of reference data

    Returns
    -------
    scaled : numpy.array
        Scaled timeseries with scaled edges
    '''

    # calculate scaling slope and new reference points at edges
    parms_lo, parms_hi, perc_ref = derive_edge_parameters(
        src=src, ref=ref, perc_src=perc_src, perc_ref=perc_ref)

    # find indexes of edge values in source data
    ids_lo = np.where(src <= perc_src[1])
    ids_hi = np.where(src >= perc_src[-2])

    # replace in new array
    inter = sc_int.InterpolatedUnivariateSpline(perc_src, perc_ref, k=1)
    scaled_edges = inter(src)
    scaled[ids_lo] = scaled_edges[ids_lo]
    scaled[ids_hi] = scaled_edges[ids_hi]

    return scaled


def resize_percentiles(in_data, percentiles, minobs):
    '''
    Shrinks bin size of percentiles if not enough data is available

    Parameters
    ----------
    in_data : numpy.array
        Input array.
    percentiles : list-like
        list of percentiles.
    minobs : int
        Minimum desired number of observations in a bin.

    Returns
    -------
    np.array
        resized percentiles.

    '''
    n = len(in_data)
    minbinsize = np.min(np.diff(percentiles))

    if n * minbinsize / 100 < minobs:
        warnings.warn("The bins have been resized")

        nbins = np.int32(np.floor(n / minobs))
        if nbins == 0:
            nbins = 1
        elif nbins > len(percentiles) - 1:
            nbins = len(percentiles) - 1

        return np.arange(nbins + 1, dtype=np.float64) / nbins * 100

    return percentiles
