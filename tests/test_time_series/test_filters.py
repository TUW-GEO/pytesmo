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
Tests for the time series filters
'''
import pytesmo.time_series.filters as filters
import numpy as np


def test_exp_filter():
    """
    Test exponential filter
    """
    test_jd = np.arange(10, dtype=np.double)
    test_data = np.array(
        [1, 2, 3, 4, -999999.0, 6, 7, 8, 9, np.nan], dtype=np.double)

    filtered = filters.exp_filter(test_data, test_jd, ctime=5)

    np.testing.assert_allclose(filtered, [1., 1.37915242, 1.89218092,
                                          2.4798615, np.nan, 3.51391292,
                                          4.43441963, 5.30393553, 6.15217733,
                                          np.nan])


def test_boxcar_filter():
    """
    Test boxcar filter
    """
    test_jd = np.arange(10, dtype=np.double)
    test_data = np.array(
        [1, 2, 3, 4, -999999.0, 6, 7, 8, 9, np.nan], dtype=np.double)

    filtered = filters.boxcar_filter(test_data, test_jd, window=5)

    np.testing.assert_allclose(filtered, [2., 2.5, 2.5, 3.75, np.nan, 6.25,
                                          7.5, 7.5, 8., np.nan])
