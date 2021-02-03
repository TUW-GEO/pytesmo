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

Created on Tue Nov  3 14:56:50 2015

@author: christoph.paulik@geo.tuwien.ac.at
'''

from pytesmo.utils import ml_percentile
from pytesmo.utils import interp_uniq
from pytesmo.utils import ensure_iterable
from pytesmo.utils import unique_percentiles_interpolate
from pytesmo.utils import unique_percentiles_beta
from pytesmo.utils import resize_percentiles
from pytesmo.utils import scale_edges
from pytesmo.utils import derive_edge_parameters
import numpy as np
import numpy.testing as nptest


def test_ml_percentile():
    """
    Test the percentile implementation that is used in Matlab.
    """

    arr1 = np.array([1, 1, 1, 2, 2, 2, 5, 5, 6, 10, 10, 10, 10])
    percentiles = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
    perc_should = [1.0, 1.0, 1.0, 1.1, 2.0, 2.0, 5.0, 5.3, 8.4, 10., 10., 10.,
                   10.]
    perc = ml_percentile(arr1, percentiles)
    nptest.assert_almost_equal(perc, perc_should)


def test_interp_unique():
    """
    test iterative filling of array
    """

    arr1 = np.array([1, 1, 1, 2, 2, 2, 5, 5, 6, 10, 10, 10, 10])
    percentiles = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
    p = ml_percentile(arr1, percentiles)
    src_perc = interp_uniq(p)
    assert len(p) == len(src_perc)

    nptest.assert_almost_equal(src_perc, [1., 1.025, 1.05, 1.1, 1.55, 3.275,
                                          5., 5.3, 8.4, 9.2, 9.6, 9.8, 10.])


def test_unique_percentile_interpolation():
    """
    test generation of unique percentile values
    by interpolation or order k
    """

    arr1 = np.array([1, 1, 1, 2, 2, 2, 5, 5, 6, 10, 10, 10, 10])
    percentiles = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
    p = ml_percentile(arr1, percentiles)
    src_perc = unique_percentiles_interpolate(p,
                                              percentiles=percentiles)
    assert len(p) == len(src_perc)

    nptest.assert_almost_equal(src_perc, [1.,   1.025,   1.05,   1.1,
                                          2.,   3.5,   5.,   5.3,
                                          8.4,   8.93333333,   9.46666667,   9.73333333,  10.])


def test_unique_percentile_beta():
    """
    test generation of unique percentile values
    by fitting CDF of a beta distribution
    """

    arr1 = np.array([1, 1, 1, 2, 2, 2, 5, 5, 6, 10, 10, 10, 10])
    percentiles = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
    p = ml_percentile(arr1, percentiles)
    src_perc = unique_percentiles_beta(p,
                                       percentiles=percentiles)
    assert len(p) == len(src_perc)

    nptest.assert_almost_equal(src_perc, [1.,   1.00013305,   1.00371443,   1.08957949,
                                          1.50096583,   2.50963215,   4.18025716,   6.24205978,
                                          8.16856852,   9.45324093,   9.94854144,   9.99597975,  10.],
                               decimal=5)


def test_ensure_iterable():
    el = 1
    new_el = ensure_iterable(el)
    assert new_el == [el]


def test_ensure_iterable_string():
    el = 'test'
    new_el = ensure_iterable(el)
    assert new_el == [el]
    
def test_derive_edge_parameters():
    """
    assert the result types for the edge parameters
    """
    src = np.linspace(-1,1,1000)
    ref = src*0.5
    percentiles = np.linspace(0,100,100)
    perc_src = ml_percentile(src, percentiles)
    perc_ref = ml_percentile(ref, percentiles)
    
    a,b,c = derive_edge_parameters(src=src, ref=ref,
                                   perc_src=perc_src, perc_ref=perc_ref)
    
    assert (type(a) is tuple) & (type(b) is tuple) & (type(c) is np.ndarray)
    
def test_scale_edges():
    """
    test that the edge values decrease to match a timeseries with smaller values
    """
    scaled = np.linspace(-1,1,1000)
    src = np.linspace(-1,1,1000)
    ref = scaled*0.5
    percentiles = np.linspace(0,100,100)
    perc_src = ml_percentile(src, percentiles)
    perc_ref = ml_percentile(ref, percentiles)
    
    edge_scaled = scale_edges(scaled=scaled,
                              src=src, ref=ref,
                              perc_src=perc_src,
                              perc_ref=perc_ref)
    test_low = np.abs(edge_scaled[:9]) < np.abs(src[:9])
    test_high = np.abs(edge_scaled[990:]) < np.abs(src[990:])
    
    assert np.all(test_low) and np.all(test_high)
    
def test_resize_percentiles():
    """
    test that the number of bins respects the 20 minimum obs. per bin
    """
    in_data = np.arange(100)
    minobs = 20
    percentiles = np.linspace(0,100,10)
    
    new_p = resize_percentiles(in_data = in_data,
                               percentiles = percentiles,
                               minobs=minobs)
    
    assert len(new_p) == 6
