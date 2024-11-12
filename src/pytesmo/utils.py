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
    Decorator for class methods or functions to mark them as deprecated.
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
            warnings.warn(
                default_msg if message is None else message,
                category=DeprecationWarning,
                stacklevel=2)

            return src(*args, **kwargs)

        return new_func

    return decorator


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
