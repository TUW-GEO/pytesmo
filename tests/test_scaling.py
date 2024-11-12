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
Created on Tue Nov  3 09:53:23 2015

Module for testing the scaling module
'''

import pytesmo.scaling as scaling
import numpy as np
import pandas as pd
import numpy.testing as nptest
import pytest

scaling_methods = list(scaling.get_scaling_method_lut().keys())


def test_mean_std_scaling():
    # use a random sample from a standard distribution
    n = 1000
    x = np.random.normal(0, 0.5, n)
    y = np.arange(n)

    o = scaling.mean_std(y, x)
    nptest.assert_almost_equal(np.std(x), np.std(o))
    nptest.assert_almost_equal(np.mean(x), np.mean(o))


def test_min_max_scaling():
    # use a random sample from a standard distribution
    n = 1000
    x = np.random.normal(0, 0.5, n)
    y = np.arange(n)

    o = scaling.min_max(y, x)
    nptest.assert_almost_equal(np.min(x), np.min(o))
    nptest.assert_almost_equal(np.max(x), np.max(o))


@pytest.mark.parametrize('method', scaling_methods)
def test_scaling_method(method):
    """
    Very simple test for linear regression.
    Could probably be done better.
    """

    # two linear functions should be matched onto each other
    n = 1000
    x = np.arange(n)
    y = np.arange(n) * 0.5

    o = getattr(scaling, method)(y, x)

    nptest.assert_almost_equal(x, o)


@pytest.mark.parametrize('method', scaling_methods)
def test_scaling_kwargs(method):
    """
    Tests that kwargs are passed correctly for every scaling method
    """

    # two linear functions should be matched onto each other
    n = 1000
    x = np.arange(n)
    y = np.arange(n) * 0.5

    kwargs = {'linear_edge_scaling': True,
              'minobs': 20,
              'max_val': 950,
              'min_val': 50}
    getattr(scaling, method)(y, x, **kwargs)


@pytest.mark.parametrize('method', scaling_methods)
def test_scale(method):
    n = 1000
    x = np.arange(n)
    y = np.arange(n) * 0.5

    df = pd.DataFrame({'x': x, 'y': y}, columns=['x', 'y'])
    df_scaled = scaling.scale(df,
                              method=method,
                              reference_index=0)
    nptest.assert_almost_equal(df_scaled['x'].values,
                               df_scaled['y'].values)


@pytest.mark.parametrize('method', ['non_existing_method'])
def test_scale_error(method):
    n = 1000
    x = np.arange(n)
    y = np.arange(n) * 0.5

    df = pd.DataFrame({'x': x, 'y': y}, columns=['x', 'y'])
    with pytest.raises(KeyError):
        df_scaled = scaling.scale(df,
                                  method=method,
                                  reference_index=0)
        nptest.assert_almost_equal(df_scaled['x'].values,
                                   df_scaled['y'].values)


@pytest.mark.parametrize('method', ['non_existing_method'])
def test_add_scale_error(method):
    n = 1000
    x = np.arange(n, dtype=float)
    y = np.arange(n) * 0.5

    df = pd.DataFrame({'x': x, 'y': y}, columns=['x', 'y'])
    with pytest.raises(KeyError):
        df_scaled = scaling.add_scaled(df, method=method)
        nptest.assert_almost_equal(df_scaled['y'].values,
                                   df_scaled['x_scaled_' + method].values)


@pytest.mark.parametrize('method', scaling_methods)
def test_add_scale(method):
    n = 1000
    x = np.arange(n, dtype=float)
    y = np.arange(n) * 0.5

    df = pd.DataFrame({'x': x, 'y': y}, columns=['x', 'y'])
    df_scaled = scaling.add_scaled(df, method=method)
    # test the scaling the other way round
    df_scaled2 = scaling.add_scaled(df, method=method,
                                    label_in='y',
                                    label_scale='x')
    nptest.assert_almost_equal(df_scaled['y'].values,
                               df_scaled['x_scaled_' + method].values)

    nptest.assert_almost_equal(df_scaled2['x'].values,
                               df_scaled2['y_scaled_' + method].values)


def test_linreg_with_nan():
    n = 1000
    x = np.arange(float(n))
    y = np.arange(float(n)) * 0.5

    x[0:10] = np.nan

    df = pd.DataFrame(data={'x': x, 'y': y})

    slope, inter = scaling.linreg_params(
        df.dropna()['x'].values, df.dropna()['y'].values
    )
    df['x'] = scaling.linreg_stored_params(df['x'].values, slope, inter)

    nptest.assert_almost_equal(df.loc[10:, 'x'].values,
                               df.loc[10:, 'y'].values)

    assert (df.index.size == n)


@pytest.mark.filterwarnings(
    "ignore:Too few percentiles for chosen k.:UserWarning")
def test_single_percentile_data():
    n = 1000
    x = np.arange(n, dtype=float)
    y = np.ones(n)

    s = scaling.cdf_match(y, x)
    nptest.assert_almost_equal(s, np.full_like(s, np.nan))
