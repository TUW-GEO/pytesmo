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

"""
Tests for the temporal matching module
Created on Wed Jul  8 19:37:14 2015
"""


from copy import deepcopy
from datetime import datetime
import numpy as np
import numpy.testing as nptest
import pandas as pd
import pytesmo.temporal_matching as tmatching
import pytest


def test_df_match_borders():
    """
    Border values can be problematic for temporal matching.

    See issue #51
    """

    ref_df = pd.DataFrame(
        {"data": np.arange(5)},
        index=pd.date_range(datetime(2007, 1, 1, 0), "2007-01-05", freq="D"),
    )
    match_df = pd.DataFrame(
        {"matched_data": np.arange(5)},
        index=[
            datetime(2007, 1, 1, 9),
            datetime(2007, 1, 2, 9),
            datetime(2007, 1, 3, 9),
            datetime(2007, 1, 4, 9),
            datetime(2007, 1, 5, 9),
        ],
    )
    with pytest.deprecated_call():
        matched = tmatching.df_match(ref_df, match_df)

    nptest.assert_allclose(
        np.array([0.375, 0.375, 0.375, 0.375, 0.375]), matched.distance.values
    )
    nptest.assert_allclose(np.arange(5), matched.matched_data)


def test_df_match_match_on_window_border():
    """
    test matching if a value lies exactly on the window border.
    """

    ref_df = pd.DataFrame(
        {"data": np.arange(5)},
        index=pd.date_range(datetime(2007, 1, 1, 0), "2007-01-05", freq="D"),
    )
    match_df = pd.DataFrame(
        {"matched_data": np.arange(4)},
        index=[
            datetime(2007, 1, 1, 9),
            datetime(2007, 1, 2, 9),
            datetime(2007, 1, 3, 12),
            datetime(2007, 1, 5, 9),
        ],
    )
    with pytest.deprecated_call():
        matched = tmatching.df_match(ref_df, match_df, window=0.5)

    nptest.assert_allclose(
        np.array([0.375, 0.375, 0.5, -0.5, 0.375]), matched.distance.values
    )
    nptest.assert_allclose([0, 1, 2, 2, 3], matched.matched_data)

    # test asym_window keyword
    with pytest.deprecated_call():
        matched = tmatching.df_match(
            ref_df, match_df, window=0.5, asym_window="<="
        )

    nptest.assert_allclose(
        np.array([0.375, 0.375, 0.5, np.nan, 0.375]), matched.distance.values
    )
    nptest.assert_allclose([0, 1, 2, np.nan, 3], matched.matched_data)

    with pytest.deprecated_call():
        matched = tmatching.df_match(
            ref_df, match_df, window=0.5, asym_window=">="
        )

    nptest.assert_allclose(
        np.array([0.375, 0.375, np.nan, -0.5, 0.375]), matched.distance.values
    )
    nptest.assert_allclose([0, 1, np.nan, 2, 3], matched.matched_data)


def test_df_match_borders_unequal_query_points():
    """
    Border values can be problematic for temporal matching.

    See issue #51
    """

    ref_df = pd.DataFrame(
        {"data": np.arange(5)},
        index=pd.date_range(datetime(2007, 1, 1, 0), "2007-01-05", freq="D"),
    )
    match_df = pd.DataFrame(
        {"matched_data": np.arange(4)},
        index=[
            datetime(2007, 1, 1, 9),
            datetime(2007, 1, 2, 9),
            datetime(2007, 1, 4, 9),
            datetime(2007, 1, 5, 9),
        ],
    )
    with pytest.deprecated_call():
        matched = tmatching.df_match(ref_df, match_df)

    nptest.assert_allclose(
        np.array([0.375, 0.375, -0.625, 0.375, 0.375]), matched.distance.values
    )
    nptest.assert_allclose(np.array([0, 1, 1, 2, 3]), matched.matched_data)


def test_matching():
    """
    test matching function
    """
    data = np.arange(5.0)
    data[3] = np.nan

    ref_df = pd.DataFrame(
        {"data": data},
        index=pd.date_range(datetime(2007, 1, 1, 0), "2007-01-05", freq="D"),
    )
    match_df = pd.DataFrame(
        {"matched_data": np.arange(5)},
        index=[
            datetime(2007, 1, 1, 9),
            datetime(2007, 1, 2, 9),
            datetime(2007, 1, 3, 9),
            datetime(2007, 1, 4, 9),
            datetime(2007, 1, 5, 9),
        ],
    )
    with pytest.deprecated_call():
        matched = tmatching.matching(ref_df, match_df)

    nptest.assert_allclose(np.array([0, 1, 2, 4]), matched.matched_data)
    assert len(matched) == 4


def test_matching_series():
    """
    test matching function with pd.Series as input
    """
    data = np.arange(5.0)
    data[3] = np.nan

    ref_ser = pd.Series(
        data,
        index=pd.date_range(datetime(2007, 1, 1, 0), "2007-01-05", freq="D"),
    )
    match_ser = pd.Series(
        np.arange(5),
        index=[
            datetime(2007, 1, 1, 9),
            datetime(2007, 1, 2, 9),
            datetime(2007, 1, 3, 9),
            datetime(2007, 1, 4, 9),
            datetime(2007, 1, 5, 9),
        ],
        name="matched_data",
    )

    with pytest.deprecated_call():
        matched = tmatching.matching(ref_ser, match_ser)

    nptest.assert_allclose(np.array([0, 1, 2, 4]), matched.matched_data)
    assert len(matched) == 4


#############################################################################
# Tests for new implementation
#############################################################################


@pytest.fixture
def test_data():
    """
    Test data for temporal matching.

    The test frames have modified time indices:
    - shifted by 3 hours
    - shifted by 7 hours
    - shifted by 3 hours, and in a timezone such that displayed numbers are >6
      hours apart
    - shifted by 7 hours, and in a timezone such that displayed numbers are <6
      hours apart
    - randomly shifted by a value between -12h and 12h
    - same as above, but with some dropped values
    - shifted by 3 hours, with duplicates

    Returns
    -------
    ref_frame : pd.DataFrame
        Reference data frame
    test_frames : dict of pd.DataFrame
        Dictionary of data frames, keys are: "shifted_3", "shifted_7",
        "shifted_3_asia", "shifted_7_us", "random_shift", "duplicates".
    expected_nan : dict of np.ndarray
        Dictionary with same keywords as `test_frames`, each entry is a mask
        indicating where NaNs are expected (i.e. no matching was taking place)
    """
    # the reference date range
    ref_dr = pd.date_range("1970", "2020", freq="D", tz="UTC")

    test_dr = {}
    test_dr["shifted_3"] = ref_dr + pd.Timedelta(3, "h")
    test_dr["shifted_7"] = ref_dr + pd.Timedelta(7, "h")
    test_dr["shifted_3_asia"] = test_dr["shifted_3"].tz_convert(
        "Asia/Yekaterinburg"
    )
    test_dr["shifted_7_us"] = test_dr["shifted_7"].tz_convert("US/Eastern")

    # random shifts
    random_hours = np.random.uniform(-12.0, 12.0, len(ref_dr))
    random_mask = np.abs(random_hours) > 6
    dr_random_shift = ref_dr + pd.to_timedelta(random_hours, "h")
    test_dr["random_shift"] = dr_random_shift

    # missing data
    drop_mask = np.zeros(len(ref_dr), dtype=bool)
    drop_mask[100:200] = True
    dr_random_shift = dr_random_shift[~drop_mask]
    test_dr["missing"] = dr_random_shift
    missing_mask = random_mask | drop_mask

    # with duplicates
    test_dr["duplicates"] = deepcopy(test_dr["shifted_3"])
    duplicates_mask = np.zeros(len(ref_dr), dtype=bool)
    for idx in np.random.randint(0, len(test_dr["duplicates"]) - 1, 5):
        test_dr["duplicates"].values[idx] = test_dr["duplicates"].values[
            idx + 1
        ]
        duplicates_mask[idx] = True

    # setting up dataframes
    test_frames = {
        key: pd.DataFrame(
            np.random.randn(len(test_dr[key]), 3), index=test_dr[key]
        )
        for key in test_dr
    }
    ref_frame = pd.DataFrame(np.random.randn(len(ref_dr), 3), index=ref_dr)

    # mask for where we expect nans in the output
    all_nan = np.ones(len(ref_dr), dtype=bool)
    expected_nan = {
        "shifted_3": ~all_nan,
        "shifted_7": all_nan,
        "shifted_3_asia": ~all_nan,
        "shifted_7_us": all_nan,
        "random_shift": random_mask,
        "missing": missing_mask,
        "duplicates": duplicates_mask,
    }
    return ref_frame, test_frames, expected_nan


def setup_data(data, key):
    """Returns only relevant data of test_data for given key"""
    return data[0], data[1][key], data[2][key]


def compare_with_nan(a, b):
    return (a == b) | (np.isnan(a) & np.isnan(b))


def assert_equal_except_nan(res, ref, nan_mask, index_shifted=False):
    expected_nan_idx = nan_mask.nonzero()[0]
    expected_nonan_idx = (~nan_mask).nonzero()[0]
    # using column zero here, all should be the same
    nan_idx = np.isnan(res.iloc[:, 0].values).nonzero()[0]
    nonan_idx = (~np.isnan(res.iloc[:, 0].values)).nonzero()[0]
    assert len(expected_nan_idx) == len(nan_idx)
    if len(nan_idx) > 0:
        assert np.all(nan_idx == expected_nan_idx)
    if len(nonan_idx) > 0 and not index_shifted:
        assert np.all(nonan_idx == expected_nonan_idx)
        assert np.all(
            res.iloc[nonan_idx, 0].values == ref.iloc[nonan_idx, 0].values
        )


@pytest.mark.parametrize(
    "key",
    [
        "shifted_3",
        "shifted_7",
        "shifted_7_us",
        "shifted_3_asia",
        "random_shift",
    ],
)
def test_collocation_nearest_neighbour(test_data, key):
    ref_frame, test_frame, expected_nan = setup_data(test_data, key)
    res = tmatching.temporal_collocation(
        ref_frame, test_frame, pd.Timedelta(6, "h")
    )
    assert_equal_except_nan(res, test_frame, expected_nan)


@pytest.mark.parametrize("key", ["missing", "duplicates"])
def test_collocation_missing_duplicates(test_data, key):
    ref_frame, test_frame, expected_nan = setup_data(test_data, key)
    res = tmatching.temporal_collocation(
        ref_frame,
        test_frame,
        pd.Timedelta(6, "h"),
    )
    # indices of test_frame are shifted w.r.t expected_nan, therefore we can't
    # compare values
    assert_equal_except_nan(res, test_frame, expected_nan, index_shifted=True)


@pytest.mark.parametrize("key", ["shifted_3"])
def test_collocation_window(test_data, key):
    ref_frame, test_frame, expected_nan = setup_data(test_data, key)
    res = tmatching.temporal_collocation(
        ref_frame, test_frame, 6 / 24, dropduplicates=True
    )
    assert_equal_except_nan(res, test_frame, expected_nan, index_shifted=True)


@pytest.mark.parametrize("key", ["shifted_3"])
def test_collocation_input(test_data, key):
    ref_frame, test_frame, expected_nan = setup_data(test_data, key)

    no_timezone = pd.date_range("1970", "2020", freq="D")
    # test with series and index:
    for ref in [ref_frame[0], ref_frame.index, no_timezone]:
        res = tmatching.temporal_collocation(
            ref, test_frame, pd.Timedelta(6, "h")
        )
        assert_equal_except_nan(res, test_frame, expected_nan)


@pytest.mark.parametrize(
    "key",
    [
        "shifted_3",
        "shifted_7",
        "shifted_7_us",
        "shifted_3_asia",
        "random_shift",
    ],
)
def test_collocation_dropna(test_data, key):
    ref_frame, test_frame, expected_nan = setup_data(test_data, key)
    res = tmatching.temporal_collocation(
        ref_frame, test_frame, pd.Timedelta(6, "h"), dropna=True
    )
    expected_nonan_idx = (~expected_nan).nonzero()[0]
    assert np.all(test_frame.iloc[expected_nonan_idx, :].values == res.values)


@pytest.mark.parametrize(
    "key",
    [
        "shifted_3",
        "shifted_7",
        "shifted_7_us",
        "shifted_3_asia",
        "random_shift",
    ],
)
def test_collocation_flag(test_data, key):
    ref_frame, test_frame, expected_nan = setup_data(test_data, key)
    flag = np.random.choice([True, False], len(ref_frame))

    # with array
    res = tmatching.temporal_collocation(
        ref_frame,
        test_frame,
        pd.Timedelta(6, "h"),
        flag=flag,
    )

    compare_with_nan(
        res.iloc[:, 0].values[~flag], test_frame.iloc[:, 0].values[~flag]
    )
    assert np.all(np.isnan(res.values[:, 0][flag]))

    # with array, using invalid as replacement
    res = tmatching.temporal_collocation(
        ref_frame,
        test_frame,
        pd.Timedelta(6, "h"),
        flag=flag,
        use_invalid=True,
    )
    compare_with_nan(res.iloc[:, 0].values, test_frame.iloc[:, 0].values)

    # with dataframe
    test_frame["flag"] = flag
    res = tmatching.temporal_collocation(
        ref_frame,
        test_frame,
        pd.Timedelta(6, "h"),
        flag="flag",
    )
    compare_with_nan(
        res.iloc[:, 0].values[~flag], test_frame.iloc[:, 0].values[~flag]
    )
    assert np.all(np.isnan(res.iloc[:, 0].values[flag]))


# using only shifted_3, because comparison won't work when there are nans
@pytest.mark.parametrize("key", ["shifted_3"])
def test_return_index(test_data, key):
    ref_frame, test_frame, expected_nan = setup_data(test_data, key)
    res = tmatching.temporal_collocation(
        ref_frame, test_frame, pd.Timedelta(6, "h"), return_index=True
    )
    assert_equal_except_nan(res, test_frame, expected_nan)
    assert np.all(test_frame.index.values == res["index_other"].values)


# using only shifted_3, because comparison won't work when there are nans
@pytest.mark.parametrize("key", ["shifted_3", "shifted_7"])
def test_return_distance(test_data, key):
    ref_frame, test_frame, expected_nan = setup_data(test_data, key)
    res = tmatching.temporal_collocation(
        ref_frame, test_frame, pd.Timedelta(6, "h"), return_distance=True
    )
    assert_equal_except_nan(res, test_frame, expected_nan)
    if key == "shifted_3":
        assert np.all(res["distance_other"] == pd.Timedelta(3, "h"))
    if key == "shifted_7":
        assert np.all(np.isnan(res["distance_other"]))


def test_timezone_handling():
    # Issue #150
    data = np.arange(5.0)
    data[3] = np.nan

    match_df = pd.DataFrame(
        {"matched_data": data},
        index=pd.date_range(
            datetime(2007, 1, 1, 0), "2007-01-05", freq="D", tz="UTC"
        ),
    )
    index = pd.DatetimeIndex(
        [
            datetime(2007, 1, 1, 9),
            datetime(2007, 1, 2, 9),
            datetime(2007, 1, 3, 9),
            datetime(2007, 1, 4, 9),
            datetime(2007, 1, 5, 9),
        ]
    ).tz_localize("utc")
    ref_df = pd.DataFrame({"data": np.arange(5)}, index=index)
    matched = tmatching.temporal_collocation(
        ref_df,
        match_df,
        pd.Timedelta(12, "h"),
        dropna=True,
    )

    nptest.assert_allclose(np.array([0, 1, 2, 4]), matched.matched_data)
    assert len(matched) == 4


def test_warning_on_no_match(test_data):
    # Issue #152
    ref_frame, test_frame, expected_nan = setup_data(test_data, "shifted_7")
    with pytest.warns(UserWarning):
        tmatching.temporal_collocation(
            ref_frame, test_frame, pd.Timedelta(6, "h"), checkna=True
        )


def test_combined_matching():
    index = pd.DatetimeIndex([datetime(2007, 1, i + 1, 0) for i in range(10)])
    ref = pd.DatetimeIndex([datetime(2007, 1, i + 1, 5) for i in range(10)])

    data = {
        "data1": np.random.randn(10),
        "data2": np.random.randn(10),
        "missing": np.random.randn(10),
    }
    data["missing"][2] = np.nan
    frames = {
        name: pd.DataFrame({name: data[name]}, index=index) for name in data
    }

    # everything together
    merged = tmatching.combined_temporal_collocation(
        ref,
        (frames[name] for name in frames),
        pd.Timedelta(6, "h"),
        combined_dropna=False,
    )

    assert len(merged) == 10
    for name in frames:
        assert name in merged.columns
        nptest.assert_equal(
            merged[name].values.ravel(), frames[name].values.ravel()
        )

    # test with dropna but not combined_dropna
    merged = tmatching.combined_temporal_collocation(
        ref,
        (frames[name] for name in frames),
        pd.Timedelta(6, "h"),
        combined_dropna=False,
        dropna=True,
    )

    assert len(merged) == 10
    for name in frames:
        assert name in merged.columns
        nptest.assert_equal(
            merged[name].values.ravel(), frames[name].values.ravel()
        )

    # test with combined_dropna
    merged = tmatching.combined_temporal_collocation(
        ref,
        (frames[name] for name in frames),
        pd.Timedelta(6, "h"),
        combined_dropna="any",
        dropna=True,
    )

    assert len(merged) == 9
    for name in frames:
        assert name in merged.columns
        nptest.assert_equal(
            merged[name].values.ravel()[2:], frames[name].values.ravel()[3:]
        )

    # test with 2d-dataframe
    df2d = pd.DataFrame(
        {"2d1": np.random.randn(10), "2d2": np.random.randn(10)}, index=index
    )
    merged = tmatching.combined_temporal_collocation(
        ref,
        (frames["missing"], df2d),
        pd.Timedelta(6, "h"),
        combined_dropna=False,
    )
    assert len(merged) == 10

    # test without match
    for comb_drop in [True, False]:
        merged = tmatching.combined_temporal_collocation(
            ref,
            (frames["missing"], df2d),
            pd.Timedelta(1, "h"),
            combined_dropna=comb_drop,
            dropna=True,
        )
        assert len(merged) == 0


def test_timezone_warning():
    dr = pd.date_range("2000-01-01", "2000-01-31", freq="D")
    dr_berlin = pd.date_range(
        "2000-01-01", "2000-01-31", freq="D", tz="Europe/Berlin"
    )
    n = len(dr)
    with pytest.warns(UserWarning, match="No timezone given"):
        matched = tmatching.temporal_collocation(
            pd.DataFrame(np.random.randn(n), index=dr),
            pd.DataFrame(np.random.randn(n), index=dr_berlin),
            pd.Timedelta(6, "h"),
        )
        assert str(matched.index.tz) == "Europe/Berlin"


def test_combined_timezones():
    dr = pd.date_range("2000-01-01", "2000-01-31", freq="D")
    dr_utc = pd.date_range("2000-01-01", "2000-01-31", freq="D", tz="UTC")
    dr_berlin = pd.date_range(
        "2000-01-01", "2000-01-31", freq="D", tz="Europe/Berlin"
    )
    n = len(dr)

    # test timezone naive
    merged = tmatching.combined_temporal_collocation(
        pd.DataFrame(np.random.randn(n), index=dr),
        (
            pd.DataFrame(np.random.randn(n), index=dr),
            pd.DataFrame(np.random.randn(n), index=dr),
        ),
        pd.Timedelta(6, "h"),
        add_ref_data=True,
    )
    assert merged.index.tz is None

    # test with same timezone
    merged = tmatching.combined_temporal_collocation(
        pd.DataFrame(np.random.randn(n), index=dr_berlin),
        (
            pd.DataFrame(np.random.randn(n), index=dr_berlin),
            pd.DataFrame(np.random.randn(n), index=dr_berlin),
        ),
        pd.Timedelta(6, "h"),
        add_ref_data=True,
    )
    assert str(merged.index.tz) == "Europe/Berlin"

    # test with missing timezone
    with pytest.warns(UserWarning, match="No timezone given"):
        merged = tmatching.combined_temporal_collocation(
            pd.DataFrame(np.random.randn(n), index=dr),
            (
                pd.DataFrame(np.random.randn(n), index=dr_berlin),
                pd.DataFrame(np.random.randn(n), index=dr),
            ),
            pd.Timedelta(6, "h"),
            add_ref_data=True,
        )
        assert str(merged.index.tz) == "Europe/Berlin"

    # test with different timezones and no ref timezone
    with pytest.warns(UserWarning) as warn_record:
        merged = tmatching.combined_temporal_collocation(
            pd.DataFrame(np.random.randn(n), index=dr),
            (
                pd.DataFrame(np.random.randn(n), index=dr_berlin),
                pd.DataFrame(np.random.randn(n), index=dr_utc),
            ),
            pd.Timedelta(6, "h"),
            add_ref_data=True,
        )
        assert str(merged.index.tz) == "UTC"
    assert len(warn_record) == 3
    assert "No timezone given" in warn_record[0].message.args[0]
    assert "Europe/Berlin" in warn_record[0].message.args[0]
    assert "No timezone given" in warn_record[1].message.args[0]
    assert "UTC" in warn_record[1].message.args[0]
    assert "mixed timezones" in warn_record[2].message.args[0]

    # test with different timezones and ref timezone
    merged = tmatching.combined_temporal_collocation(
        pd.DataFrame(np.random.randn(n), index=dr_berlin),
        (
            pd.DataFrame(np.random.randn(n), index=dr_berlin),
            pd.DataFrame(np.random.randn(n), index=dr_utc),
        ),
        pd.Timedelta(6, "h"),
        add_ref_data=True,
    )
    assert str(merged.index.tz) == "Europe/Berlin"


def test_resample_mean():

    times = np.arange(10, dtype=float)
    vals = np.arange(10, dtype=float)

    target_times = np.asarray([3, 7, 10, 12], dtype=float)
    window = 3.0

    expected = np.asarray([3, 7, 9, np.nan])

    resampled = tmatching.resample_mean(times, vals, target_times, window)

    def repeat(x, n):
        return np.repeat(x[np.newaxis, :], n, axis=0)

    newvals = repeat(vals, 12)
    new_expected = repeat(expected, 12)
    resampled = tmatching.resample_mean(times, newvals, target_times, window)
    np.testing.assert_equal(new_expected, resampled)


def test_mean_collocation():
    dr = pd.date_range("1970", "2020", freq="D", tz="UTC")
    other = pd.DataFrame(
        np.vstack(
            (np.arange(len(dr), dtype=float), np.arange(len(dr), dtype=float))
        ).T,
        index=dr,
    )
    # window is actually half window in temporal_collocation
    window = pd.Timedelta(1.5, "D")

    dateidx = list(range(0, len(dr) - 3, 3))
    expected = list(range(0, len(dr) - 3, 3))
    expected[0] = 0.5
    ref_dr = pd.DatetimeIndex([dr[i] for i in dateidx])

    resampled = tmatching.temporal_collocation(
        ref_dr, other, window, method="mean"
    )
    assert (resampled[0] == resampled[1]).all()
    assert (resampled[0] == expected).all()

    # try with Series
    s = tmatching.temporal_collocation(ref_dr, other[0], window, method="mean")
    assert (s == resampled[0]).all()
    assert isinstance(s, pd.Series)


def test_mean_collocation_missing_start_end():
    # In this test the first and last ten timestamps have no values within the
    # window. This is to make sure that indexing at the start and end works as
    # expected.

    dr = pd.date_range("2019", "2020", freq="D", tz="UTC")
    other_dr = pd.date_range("2019", "2020", freq="D", tz="UTC").values
    other_dr[0:10] += pd.Timedelta(12, "h")
    other_dr[-10:] -= pd.Timedelta(12, "h")
    other = pd.DataFrame(
        np.vstack(
            (np.arange(len(dr), dtype=float), np.arange(len(dr), dtype=float))
        ).T,
        index=other_dr,
    )

    resampled = tmatching.temporal_collocation(
        dr, other, pd.Timedelta(6, "h"), method="mean"
    )

    assert np.all(np.isnan(resampled[0:10]))
    assert np.all(np.isnan(resampled[-10:]))
    np.testing.assert_equal(other[10:-10].values, resampled[10:-10].values)

    # test with "random" data that uses a seed with similar properties
    np.random.seed(8986)
    ref = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    values = np.random.randn(len(ref), 3)
    random_shift = np.random.uniform(-12, 12, len(ref))
    random = pd.DataFrame(
        values, index=ref + pd.to_timedelta(random_shift, "h"),
        columns=list(map(lambda x: f"random_{x}", range(3)))
    )
    window = pd.Timedelta(hours=6)
    matched = tmatching.temporal_collocation(
        ref, random, window, method="mean"
    )

    should_be_nan = np.abs(random_shift) > 6
    expected = np.array(values)
    expected[should_be_nan, :] = np.nan
    np.testing.assert_equal(expected, matched)
