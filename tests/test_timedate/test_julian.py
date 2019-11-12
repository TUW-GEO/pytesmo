# Copyright (c) 2016,Vienna University of Technology,
# Department of Geodesy and Geoinformation
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#   * Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#   * Neither the name of the Vienna University of Technology, Department of
#     Geodesy and Geoinformation nor the names of its contributors may be used
#     to endorse or promote products derived from this software without specific
#     prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL VIENNA UNIVERSITY OF TECHNOLOGY, DEPARTMENT OF
# GEODESY AND GEOINFORMATION BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

'''
Tests for julian date conversion.
'''

import numpy as np
import numpy.testing as nptest
from datetime import datetime
from pytesmo.timedate.julian import julday
from pytesmo.timedate.julian import caldat
from pytesmo.timedate.julian import julian2date
from pytesmo.timedate.julian import julian2datetime
from pytesmo.timedate.julian import doy


def test_julday():
    jd = julday(5, 25, 2016, 10, 20, 11)
    jd_should = 2457533.9306828701
    nptest.assert_almost_equal(jd, jd_should)


def test_julday_arrays():
    jds = julday(np.array([5, 5]),
                 np.array([25, 25]),
                 np.array([2016, 2016]),
                 np.array([10, 10]),
                 np.array([20, 20]),
                 np.array([11, 11]))
    jds_should = np.array([2457533.93068287,
                           2457533.93068287])
    nptest.assert_almost_equal(jds, jds_should)


def test_julday_single_arrays():
    jds = julday(np.array([5]),
                 np.array([25]),
                 np.array([2016]),
                 np.array([10]),
                 np.array([20]),
                 np.array([11]))
    jds_should = np.array([2457533.93068287])
    nptest.assert_almost_equal(jds, jds_should)


def test_caldat():
    month, day, year = caldat(2457533.93068287)
    assert month == 5
    assert day == 25
    assert year == 2016


def test_caldat_array():
    month, day, year = caldat(np.array([2457533.93068287,
                                        2457533.93068287]))
    nptest.assert_almost_equal(month, np.array([5, 5]))
    nptest.assert_almost_equal(day, np.array([25, 25]))
    nptest.assert_almost_equal(year, np.array([2016, 2016]))


def test_julian2date():
    (year, month, day,
     hour, minute, second, micro) = julian2date(2457533.9306828701)
    assert type(year) == int
    assert year == 2016
    assert month == 5
    assert day == 25
    assert hour == 10
    assert minute == 20
    assert second == 10
    assert micro == 999976
    (year, month, day,
     hour, minute, second, micro) = julian2date(2454515.40972)
    assert year == 2008
    assert month == 2
    assert day == 18
    assert hour == 21
    assert minute == 49
    assert second == 59
    assert micro == 807989


def test_julian2date_single_array():
    (year, month, day,
     hour, minute, second, micro) = julian2date(np.array([2457533.9306828701]))
    assert type(year) == np.ndarray
    assert year == 2016
    assert month == 5
    assert day == 25
    assert hour == 10
    assert minute == 20
    assert second == 10
    assert micro == 999976


def test_julian2date_array():
    (year, month, day,
     hour, minute, second, micro) = julian2date(np.array([2457533.9306828701,
                                                          2457533.9306828701]))
    nptest.assert_almost_equal(year, np.array([2016, 2016]))
    nptest.assert_almost_equal(month, np.array([5, 5]))
    nptest.assert_almost_equal(day, np.array([25, 25]))
    nptest.assert_almost_equal(hour, np.array([10, 10]))
    nptest.assert_almost_equal(minute, np.array([20, 20]))
    nptest.assert_almost_equal(second, np.array([10, 10]))
    nptest.assert_almost_equal(micro, np.array([999976, 999976]))


def test_julian2datetime():
    dt = julian2datetime(2457533.9306828701)
    dt_should = datetime(2016, 5, 25, 10, 20, 10, 999976)
    assert dt == dt_should
    dt = julian2datetime(2457173.8604166666)
    dt_should = datetime(2015, 5, 31, 8, 39)
    assert dt == dt_should


def test_julian2datetime_single_array():
    dt = julian2datetime(np.array([2457533.9306828701]))
    dt_should = np.array([datetime(2016, 5, 25, 10, 20, 10, 999976)])
    assert type(dt) == np.ndarray
    assert np.all(dt == dt_should)


def test_julian2datetime_array():
    dt = julian2datetime(np.array([2457533.9306828701,
                                   2457533.9306828701]))
    dts = datetime(2016, 5, 25, 10, 20, 10, 999976)
    dt_should = np.array([dts, dts])
    assert type(dt) == np.ndarray
    assert np.all(dt == dt_should)


def test_doy():
    day_of_year = doy(1, 28)
    assert day_of_year == 28
    day_of_year = doy(2, 29)
    assert day_of_year == 31 + 29
    day_of_year = doy(3, 1, year=2004)
    assert day_of_year == 31 + 29 + 1
    # test numpy arrays as input
    days = np.array([28, 29, 1], dtype=int)
    months = np.array([1, 2, 3])
    days_of_year = doy(months, days, year=np.array([2005, 2004, 2004]))
    nptest.assert_allclose(days_of_year, np.array([28,
                                                   31 + 29,
                                                   31 + 29 + 1]))

    days_of_year = doy(months, days, year=2004)
    nptest.assert_allclose(days_of_year, np.array([28,
                                                   31 + 29,
                                                   31 + 29 + 1]))
