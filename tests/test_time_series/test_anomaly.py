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
Test for climatology and anomaly calculation.
'''
import pdb

import pandas as pd
import pandas.testing as pdt
import numpy as np

import pytesmo.time_series.anomaly as anomaly
from pytesmo.time_series.anomaly import moving_average


def test_anomaly_calc_given_climatology():

    clim = pd.Series(np.arange(366), name='clim', index=np.arange(366) + 1)
    data = pd.Series(
        np.arange(366), index=pd.date_range('2000-01-01', periods=366))
    anom_should = pd.Series(
        np.zeros(366), index=pd.date_range('2000-01-01', periods=366))
    anom = anomaly.calc_anomaly(
        data, climatology=clim, respect_leap_years=False)

    pdt.assert_series_equal(anom_should, anom, check_dtype=False)


def test_anomaly_calc_given_climatology_return_clim():

    clim = pd.Series(np.arange(366), name='clim', index=np.arange(366) + 1)
    data = pd.Series(
        np.arange(366), index=pd.date_range('2000-01-01', periods=366))
    anom_should = pd.Series(
        np.zeros(366), index=pd.date_range('2000-01-01', periods=366),
        name='anomaly')
    anom = anomaly.calc_anomaly(
        data, climatology=clim, respect_leap_years=False,
        return_clim=True)

    pdt.assert_series_equal(anom_should, anom['anomaly'], check_dtype=False)


def test_anomaly_calc_given_climatology_no_leap_year():

    clim = pd.Series(np.arange(366), name='clim', index=np.arange(366) + 1)
    data = pd.Series(
        np.arange(365), index=pd.date_range('2007-01-01', periods=365))
    anom_should = pd.Series(
        np.zeros(365), index=pd.date_range('2007-01-01', periods=365))
    anom = anomaly.calc_anomaly(
        data, climatology=clim, respect_leap_years=True)

    pdt.assert_series_equal(anom_should, anom, check_dtype=False)


def test_climatology_always_366():
    ts = pd.Series(np.sin(np.arange(366) / 366. * 2 * np.pi), index=pd.date_range(
        '2000-01-01', freq='D', periods=366))
    # remove a part of the time series
    ts['2000-02-01': '2000-02-28'] = np.nan
    ts = ts.dropna()
    clim = anomaly.calc_climatology(ts)
    assert clim.size == 366

    # this should also be the case if interpolate_leapday is set
    ts = pd.Series(np.sin(np.arange(10)), index=pd.date_range(
        '2000-01-01', freq='D', periods=10))
    clim = anomaly.calc_climatology(ts, interpolate_leapday=True)
    assert clim.size == 366


def test_monthly_climatology_always_12():
    ts = pd.Series(np.sin(np.arange(366) / 366. * 2 * np.pi), index=pd.date_range(
        '2000-01-01', freq='D', periods=366))
    # remove a part of the time series
    ts['2000-02-01': '2000-02-28'] = np.nan
    ts = ts.dropna()
    clim = anomaly.calc_climatology(ts, output_freq="month")
    assert clim.size == 12

    # this should also be the case if interpolate_leapday is set
    ts = pd.Series(np.sin(np.arange(10)), index=pd.date_range(
        '2000-01-01', freq='D', periods=10))
    # and if the window size is changed
    clim = anomaly.calc_climatology(ts, moving_avg_clim=5, output_freq="month")
    assert clim.size == 12


def test_climatology_always_366_fill():
    ts = pd.Series(np.sin(np.arange(366) / 366. * 2 * np.pi), index=pd.date_range(
        '2000-01-01', freq='D', periods=366))
    # remove a part of the time series
    ts['2000-02-01': '2000-02-28'] = np.nan
    ts = ts.dropna()
    clim = anomaly.calc_climatology(ts, fill=-1, moving_avg_clim=1)
    assert clim.size == 366
    assert clim.iloc[31] == -1


def test_climatology_closed():
    ts = pd.Series(np.arange(366), index=pd.date_range(
        '2000-01-01', freq='D', periods=366))
    # remove a part of the time series
    ts['2000-02-01': '2000-02-28'] = np.nan
    ts = ts.dropna()
    clim = anomaly.calc_climatology(ts, wraparound=True)
    assert clim.size == 366
    # test that the arange was closed during the second moving average
    assert clim.iloc[365] - 187.90 < 0.01


def test_climatology_std():
    np.random.seed(451)
    idx = pd.date_range('2000-01-01', '2010-12-31', freq='D')
    ts = pd.Series(index=idx, data=np.random.rand(len(idx)))

    # remove a part of the time series
    clim = anomaly.calc_climatology(ts, std=True)

    # use same kwargs as in the function above (default)
    ser = moving_average(ts, window_size=5, fillna=False, min_obs=1)

    # in the clac_climatology function std is computed after first smoothing
    assert clim.loc[3, 'std'] == ser.groupby(ser.index.dayofyear).std().loc[3]

    assert clim.index.size == 366
    assert 'std' in clim.columns
    assert 'climatology' in clim.columns


def test_zscores():
    np.random.seed(451)
    idx = pd.date_range('2000-01-01', '2010-12-31', freq='D')
    ts = pd.Series(index=idx, data=np.random.rand(len(idx)))
    clim = anomaly.calc_climatology(ts, std=True)
    anom = anomaly.calc_anomaly(ts, climatology=clim, return_clim=True)
    assert 'anomaly' in anom.columns
    assert 'climatology' in anom.columns
    assert 'climatology_std' in anom.columns

    zscore = anom['anomaly'] / anom['climatology_std']
    assert np.all(zscore.values[np.where(anom.anomaly < 0)[0]] < 0)
    assert np.all(zscore.values[np.where(anom.anomaly >= 0)[0]] >= 0)
