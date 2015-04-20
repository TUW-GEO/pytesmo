# Copyright (c) 2015, Vienna University of Technology,
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
# names of its contributors may be used to endorse or promote products #
# derived from this software without specific prior written permission.

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
Test for the metrics
Created on Fri Feb  6 11:25:40 2015

@author: christoph.paulik@geo.tuwien.ac.at
'''

import pytesmo.metrics as met
import numpy as np

import numpy.testing as nptest


def test_pearson_conf():
    """
    Test the person confidence interval based on
    the fisher z-transform
    """
    # first example
    n = 34
    r = 0.654
    rl, ru = met.pearson_conf(r, n, c=95)
    nptest.assert_almost_equal(rl, 0.406, decimal=3)
    nptest.assert_almost_equal(ru, 0.812, decimal=3)

    rl, ru = met.pearson_conf(r, n, c=99)
    nptest.assert_almost_equal(rl, 0.309, decimal=3)
    nptest.assert_almost_equal(ru, 0.8468, decimal=3)

    # second example
    r = 0.824
    n = 300
    rl, ru = met.pearson_conf(r, n, c=95)
    nptest.assert_almost_equal(rl, 0.784, decimal=3)
    nptest.assert_almost_equal(ru, 0.857, decimal=3)

    rl, ru = met.pearson_conf(r, n, c=99)
    nptest.assert_almost_equal(rl, 0.7697, decimal=3)
    nptest.assert_almost_equal(ru, 0.866, decimal=3)

    # test numpy arrays as input

    r = np.array([0.654, 0.824])
    n = np.array([34, 300])
    rl, ru = met.pearson_conf(r, n, c=95)
    nptest.assert_almost_equal(rl, np.array([0.406, 0.784]), decimal=3)
    nptest.assert_almost_equal(ru, np.array([0.812, 0.857]), decimal=3)


def test_bias():
    """
    Test for bias
    """
    # example 1
    x = np.arange(10)
    y = np.arange(10) + 2
    b_pred = -2
    b_obs = met.bias(x, y)

    nptest.assert_equal(b_obs, b_pred)

    # example 2
    x = np.arange(10)
    y = np.arange(20, 30)
    b_pred = 20
    b_obs = met.bias(y, x)

    nptest.assert_equal(b_obs, b_pred)


def test_aad():
    """
    Test for average absolute deviation
    """
    # example 1
    x = np.arange(10)
    y = np.arange(10) + 2
    dev_pred = 2
    dev_obs = met.aad(x, y)

    nptest.assert_equal(dev_obs, dev_pred)

    # example 2, with outlier
    x = np.arange(10)
    y = np.arange(10) + 2
    y[-1] = 201
    dev_pred = 21
    dev_obs = met.aad(x, y)

    nptest.assert_equal(dev_obs, dev_pred)


def test_mad():
    """
    Test for median absolute deviation
    """
    # example 1
    x = np.arange(10)
    y = np.arange(10) + 2
    dev_pred = 2
    dev_obs = met.mad(x, y)

    nptest.assert_equal(dev_obs, dev_pred)

    # example 2, with outlier
    x = np.arange(10)
    y = np.arange(10) + 2
    y[-1] = 201
    dev_pred = 2
    dev_obs = met.mad(x, y)

    nptest.assert_equal(dev_obs, dev_pred)


def test_rmsd():
    """
    Test for rmsd
    """
    # example 1
    x = np.arange(10)
    y = np.arange(10) + 2
    rmsd_pred = 2
    rmsd_obs = met.rmsd(x, y)

    nptest.assert_equal(rmsd_obs, rmsd_pred)

    # example 2, with outlier
    x = np.arange(10)
    y = np.arange(10) + 2
    y[-1] = 100
    rmsd_pred = np.sqrt(831)
    rmsd_obs = met.rmsd(x, y)

    nptest.assert_almost_equal(rmsd_obs, rmsd_pred, 6)


def test_rmsd_mse():
    """
    Test for rmsd and mse
    """
    # example 1
    x = np.random.randn(100)
    y = np.random.randn(100)
    rmsd_pred = met.rmsd(x, y)
    mse_pred, _, _, _ = met.mse(x, y)

    nptest.assert_equal(rmsd_pred**2, mse_pred)
