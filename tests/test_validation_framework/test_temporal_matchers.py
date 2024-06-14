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
#     to endorse or promote products derived from this software without
#     specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL VIENNA UNIVERSITY OF TECHNOLOGY, DEPARTMENT
# OF GEODESY AND GEOINFORMATION BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''
Test for temporal matchers
'''

import numpy as np
import pandas as pd
from pathlib import Path

import pytesmo.validation_framework.temporal_matchers as temporal_matchers


def test_combinatory_matcher_n2():

    n = 1000
    x = np.arange(n)
    y = np.arange(n) * 0.5
    index = pd.date_range(start="2000-01-01", periods=n, freq="D")

    df = pd.DataFrame({'x': x, 'y': y}, columns=['x', 'y'], index=index)
    df2 = pd.DataFrame({'x': x, 'y': y}, columns=['x', 'y'], index=index)
    df3 = pd.DataFrame({'x': x, 'y': y}, columns=['x', 'y'], index=index)

    df_dict = {'data1': df,
               'data2': df2,
               'data3': df3}

    temp_matcher = temporal_matchers.BasicTemporalMatching()
    matched = temp_matcher.combinatory_matcher(df_dict, 'data1')
    assert sorted(list(matched)) == sorted([('data1', 'data2'),
                                            ('data1', 'data3')])
    assert sorted(list(matched[('data1',
                                'data2')].columns)) == sorted([('data1', 'x'),
                                                               ('data1', 'y'),
                                                               ('data2', 'x'),
                                                               ('data2', 'y')])

    assert sorted(list(matched[('data1',
                                'data3')].columns)) == sorted([('data1', 'x'),
                                                               ('data1', 'y'),
                                                               ('data3', 'x'),
                                                               ('data3', 'y')])


def test_combinatory_matcher_n3():

    n = 1000
    x = np.arange(n)
    y = np.arange(n) * 0.5
    index = pd.date_range(start="2000-01-01", periods=n, freq="D")

    df = pd.DataFrame({'x': x, 'y': y}, columns=['x', 'y'], index=index)
    df2 = pd.DataFrame({'x': x, 'y': y}, columns=['x', 'y'], index=index)
    df3 = pd.DataFrame({'x': x, 'y': y}, columns=['x', 'y'], index=index)
    df4 = pd.DataFrame({'x': x, 'y': y}, columns=['x', 'y'], index=index)

    df_dict = {'data1': df,
               'data2': df2,
               'data3': df3}

    temp_matcher = temporal_matchers.BasicTemporalMatching()
    matched = temp_matcher.combinatory_matcher(df_dict, 'data1', n=3)
    assert list(matched) == [('data1', 'data2', 'data3')]
    assert sorted(list(matched[('data1',
                                'data2',
                                'data3')].columns)) == sorted([('data1', 'x'),
                                                               ('data1', 'y'),
                                                               ('data2', 'x'),
                                                               ('data2', 'y'),
                                                               ('data3', 'x'),
                                                               ('data3', 'y')])

    df_dict = {'data1': df,
               'data2': df2,
               'data3': df3,
               'data4': df4}

    temp_matcher = temporal_matchers.BasicTemporalMatching()
    matched = temp_matcher.combinatory_matcher(df_dict, 'data1', n=3)
    assert sorted(list(matched)) == sorted([('data1', 'data2', 'data3'),
                                            ('data1', 'data2', 'data4'),
                                            ('data1', 'data3', 'data4')])
    assert sorted(list(matched[('data1',
                                'data2',
                                'data3')].columns)) == sorted([('data1', 'x'),
                                                               ('data1', 'y'),
                                                               ('data2', 'x'),
                                                               ('data2', 'y'),
                                                               ('data3', 'x'),
                                                               ('data3', 'y')])


def test_add_name_to_df_columns():

    n = 10
    x = np.arange(n)
    y = np.arange(n) * 0.5
    index = pd.date_range(start="2000-01-01", periods=n, freq="D")

    df = pd.DataFrame({'x': x, 'y': y}, columns=['x', 'y'], index=index)
    df = temporal_matchers.df_name_multiindex(df, 'test')
    assert list(df.columns) == [('test', 'x'), ('test', 'y')]


def test_dfdict_combined_temporal_collocation():

    ref_dr = pd.date_range("2000", "2020", freq="YS")
    dr1 = pd.date_range("2000", "2015", freq="YS")
    dr2 = pd.date_range("2005", "2020", freq="YS")

    ref_df = pd.DataFrame({"ref": np.arange(len(ref_dr))}, index=ref_dr)
    df1 = pd.DataFrame(
        {"k1": np.arange(len(dr1)), "k2": np.arange(len(dr1))}, index=dr1
    )
    df2 = pd.DataFrame(
        {"k1": np.arange(len(dr2)), "k2": np.arange(len(dr2))}, index=dr2
    )

    dfs = {"refkey": ref_df, "df1key": df1, "df2key": df2}
    window = pd.Timedelta(days=300)

    matched = temporal_matchers.dfdict_combined_temporal_collocation(
        dfs, "refkey", 2, window=window, n=3, combined_dropna=True
    )

    # keys are the same, only refkey is missing
    key = ("refkey", "df1key", "df2key")
    assert list(matched.keys()) == [key]

    # overlap is only 11 timestamps
    assert matched[key].shape == (11, 5)

    overlap_dr = pd.date_range("2005", "2015", freq="YS")
    assert np.all(matched[key].index == overlap_dr)

    # test with ASCAT and ISMN data
    here = Path(__file__).resolve().parent
    ascat = pd.read_csv(here / "ASCAT.csv", index_col=0, parse_dates=True)
    ismn = pd.read_csv(here / "ISMN.csv", index_col=0, parse_dates=True)

    dfs = {"ASCAT": ascat[["sm"]], "ISMN": ismn[["soil_moisture"]]}
    refname = "ISMN"
    window = pd.Timedelta(12, "h")

    old_matcher = temporal_matchers.BasicTemporalMatching().combinatory_matcher
    new_matcher = temporal_matchers.make_combined_temporal_matcher(window)

    expected = old_matcher(dfs, refname, k=2, n=2)
    new = new_matcher(dfs, refname, k=2, n=2)

    key = ("ISMN", "ASCAT")
    assert list(expected.keys()) == [key]
    assert list(new.keys()) == [key]
    # We have to do an extra dropna for the old matcher, because the old
    # matcher doesn't do this by itself.
    # This is normally done within validation.py, `get_data_for_result_tuple`,
    # but since the combined matcher should exclude all data where even a
    # single entry misses (so that all only have common data) this is done
    # before in the new matcher (the combined matcher, whereas the old one is
    # the combinatory matcher)
    exp = expected[key].dropna()
    assert exp.shape == new[key].shape
    for col in new[key]:
        np.testing.assert_equal(exp[col].values, new[key][col].values)
