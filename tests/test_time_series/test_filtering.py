# Copyright (c) 2016, TU Wien
# Department of Geodesy and Geoinformation
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#   * Neither the name of the Vienna University of Technology,
#     Department of Geodesy and Geoinformation nor the
#     names of its contributors may be used to endorse or promote products
#     derived from this software without specific prior written permission.

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
Tests for the time series filtering module
'''
import pytesmo.time_series.filtering as filtering
import numpy as np
import pandas as pd


def test_moving_average():
    """
    Test moving average filter.
    """
    test_jd = np.arange(10, dtype=np.double)
    test_data = np.array(
        [1, 2, 3, 4, -999999.0, 6, 7, 8, 9, np.nan], dtype=np.double)

    ser = pd.Series(test_data, index=test_jd)

    filtered = filtering.moving_average(ser, window_size=5)

    np.testing.assert_allclose(filtered.values, [2., 2.5, 2.5, 3.75, np.nan, 6.25,
                                                 7.5, 7.5, 8., np.nan])


def test_moving_average_dt_index():
    """
    Test moving average filter with datetimeindex.
    """
    test_jd = pd.date_range(start='2000-01-01', periods=10, freq='D')
    test_data = np.array(
        [1, 2, 3, 4, -999999.0, 6, 7, 8, 9, np.nan], dtype=np.double)

    ser = pd.Series(test_data, index=test_jd)

    filtered = filtering.moving_average(ser, window_size=5)

    np.testing.assert_allclose(filtered.values, [2., 2.5, 2.5, 3.75, np.nan, 6.25,
                                                 7.5, 7.5, 8., np.nan])


def test_moving_average_size_1():
    """
    Test moving average filter with input size 1.
    """
    test_jd = np.arange(1, dtype=np.double)
    test_data = np.array(
        [1], dtype=np.double)

    ser = pd.Series(test_data, index=test_jd)

    filtered = filtering.moving_average(ser, window_size=5)

    np.testing.assert_allclose(filtered.values, [1.])


def test_moving_average_fillna():
    """
    Test moving average filter with datetimeindex.
    """
    test_jd = pd.date_range(start='2000-01-01', periods=12, freq='D')
    test_data = np.array(
        [1, 2, 3, 4, np.nan, np.nan, 8, 9, 10, np.nan, np.nan, np.nan], dtype=np.double)

    ser = pd.Series(test_data, index=test_jd)

    filtered = filtering.moving_average(ser, window_size=5, fillna=True)

    np.testing.assert_allclose(filtered.values, [2., 2.5, 2.5, 3.0, 5.0, 7.0, 9.0, 9.0, 9.0, 9.5, 10, np.nan])


def test_moving_average_min_observations():
    """
    Test moving average filter with datetimeindex.
    """
    test_jd = pd.date_range(start='2000-01-01', periods=12, freq='D')
    test_data = np.array(
        [1, 2, 3, 4, np.nan, np.nan, 8, 9, 10, np.nan, np.nan, np.nan], dtype=np.double)

    ser = pd.Series(test_data, index=test_jd)

    filtered = filtering.moving_average(ser, window_size=5, fillna=True, min_obs=3)

    np.testing.assert_allclose(filtered.values, [2., 2.5, 2.5, 3.0, 5.0, 7.0, 9.0, 9.0, 9.0, np.nan, np.nan, np.nan])

    filtered = filtering.moving_average(ser, window_size=5, fillna=True, min_obs=4)

    np.testing.assert_allclose(filtered.values, [np.nan, 2.5, 2.5, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])