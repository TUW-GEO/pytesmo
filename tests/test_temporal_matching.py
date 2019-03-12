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
Tests for the temporal matching module
Created on Wed Jul  8 19:37:14 2015
'''


import pytesmo.temporal_matching as tmatching
import pandas as pd
from datetime import datetime
import numpy as np
import numpy.testing as nptest


def test_df_match_borders():
    """
    Border values can be problematic for temporal matching.

    See issue #51
    """

    ref_df = pd.DataFrame({"data": np.arange(5)}, index=pd.date_range(datetime(2007, 1, 1, 0),
                                                                      "2007-01-05", freq="D"))
    match_df = pd.DataFrame({"matched_data": np.arange(5)},
                            index=[datetime(2007, 1, 1, 9),
                                   datetime(2007, 1, 2, 9),
                                   datetime(2007, 1, 3, 9),
                                   datetime(2007, 1, 4, 9),
                                   datetime(2007, 1, 5, 9)])
    matched = tmatching.df_match(ref_df, match_df)

    nptest.assert_allclose(
        np.array([0.375, 0.375, 0.375, 0.375, 0.375]), matched.distance.values)
    nptest.assert_allclose(np.arange(5), matched.matched_data)


def test_df_match_match_on_window_border():
    """
    test matching if a value lies exactly on the window border.
    """

    ref_df = pd.DataFrame({"data": np.arange(5)}, index=pd.date_range(datetime(2007, 1, 1, 0),
                                                                      "2007-01-05", freq="D"))
    match_df = pd.DataFrame({"matched_data": np.arange(4)},
                            index=[datetime(2007, 1, 1, 9),
                                   datetime(2007, 1, 2, 9),
                                   datetime(2007, 1, 3, 12),
                                   datetime(2007, 1, 5, 9)])
    matched = tmatching.df_match(ref_df, match_df, window=0.5)

    nptest.assert_allclose(
        np.array([0.375, 0.375, 0.5, -0.5, 0.375]), matched.distance.values)
    nptest.assert_allclose([0, 1, 2, 2, 3], matched.matched_data)

    # test asym_window keyword
    matched = tmatching.df_match(
        ref_df, match_df, window=0.5, asym_window="<=")

    nptest.assert_allclose(
        np.array([0.375, 0.375, 0.5, np.nan, 0.375]), matched.distance.values)
    nptest.assert_allclose([0, 1, 2, np.nan, 3], matched.matched_data)

    matched = tmatching.df_match(
        ref_df, match_df, window=0.5, asym_window=">=")

    nptest.assert_allclose(
        np.array([0.375, 0.375, np.nan, -0.5, 0.375]), matched.distance.values)
    nptest.assert_allclose([0, 1, np.nan, 2, 3], matched.matched_data)


def test_df_match_borders_unequal_query_points():
    """
    Border values can be problematic for temporal matching.

    See issue #51
    """

    ref_df = pd.DataFrame({"data": np.arange(5)}, index=pd.date_range(datetime(2007, 1, 1, 0),
                                                                      "2007-01-05", freq="D"))
    match_df = pd.DataFrame({"matched_data": np.arange(4)},
                            index=[datetime(2007, 1, 1, 9),
                                   datetime(2007, 1, 2, 9),
                                   datetime(2007, 1, 4, 9),
                                   datetime(2007, 1, 5, 9)])
    matched = tmatching.df_match(ref_df, match_df)

    nptest.assert_allclose(
        np.array([0.375, 0.375, -0.625, 0.375, 0.375]), matched.distance.values)
    nptest.assert_allclose(np.array([0, 1, 1, 2, 3]), matched.matched_data)


def test_matching():
    """
    test matching function
    """
    data = np.arange(5.0)
    data[3] = np.nan

    ref_df = pd.DataFrame({"data": data}, index=pd.date_range(datetime(2007, 1, 1, 0),
                                                              "2007-01-05", freq="D"))
    match_df = pd.DataFrame({"matched_data": np.arange(5)},
                            index=[datetime(2007, 1, 1, 9),
                                   datetime(2007, 1, 2, 9),
                                   datetime(2007, 1, 3, 9),
                                   datetime(2007, 1, 4, 9),
                                   datetime(2007, 1, 5, 9)])
    matched = tmatching.matching(ref_df, match_df)

    nptest.assert_allclose(np.array([0, 1, 2, 4]), matched.matched_data)
    assert len(matched) == 4


def test_matching_series():
    """
    test matching function with pd.Series as input
    """
    data = np.arange(5.0)
    data[3] = np.nan

    ref_ser = pd.Series(data, index=pd.date_range(datetime(2007, 1, 1, 0),
                                                  "2007-01-05", freq="D"))
    match_ser = pd.Series(np.arange(5),
                          index=[datetime(2007, 1, 1, 9),
                                 datetime(2007, 1, 2, 9),
                                 datetime(2007, 1, 3, 9),
                                 datetime(2007, 1, 4, 9),
                                 datetime(2007, 1, 5, 9)],
                          name='matched_data')

    matched = tmatching.matching(ref_ser, match_ser)

    nptest.assert_allclose(np.array([0, 1, 2, 4]), matched.matched_data)
    assert len(matched) == 4
