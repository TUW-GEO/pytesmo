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
                    src[
                        pos[-1]] = (src[pos[len(pos) - 2]] + src[pos[-1] + 1]) / 2.0
                elif pos[-1] == size - 1:
                    src[pos[0]] = (
                        src[pos[1]] + src[pos[0] - 1]) / 2.0
                else:
                    src[pos[0]] = (
                        src[pos[1]] + src[pos[0] - 1]) / 2.0
                    src[pos[1]] = (
                        src[pos[0]] + src[pos[1] + 1]) / 2.0
            uniq_ind = np.unique(src, return_index=True)[1]

    return src


def unique_percentiles_interpolate(perc_values,
                                   percentiles=[0, 5, 10, 30, 50,
                                                70, 90, 95, 100],
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
    uniq_ind = np.unique(perc_values, return_index=True)[1]
    if len(uniq_ind) == 1:
        uniq_ind = np.repeat(uniq_ind, 2)
    uniq_ind[-1] = len(percentiles) - 1
    uniq_perc_values = perc_values[uniq_ind]

    inter = sc_int.InterpolatedUnivariateSpline(
        np.array(percentiles)[uniq_ind],
        uniq_perc_values,
        k=k, ext=0,
        check_finite=True)
    uniq_perc_values = inter(percentiles)
    return uniq_perc_values


def unique_percentiles_beta(perc_values,
                            percentiles):
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
    min_value = np.min(perc_values)
    perc_values = perc_values - min_value
    max_value = np.max(perc_values)
    perc_values = perc_values / max_value
    percentiles = np.asanyarray(percentiles)
    percentiles = percentiles / 100.0

    p, ier = sc_opt.curve_fit(betainc,
                              percentiles,
                              perc_values)
    uniq_perc_values = sc_special.betainc(p[0], p[1], percentiles)
    uniq_perc_values = uniq_perc_values * max_value + min_value
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
