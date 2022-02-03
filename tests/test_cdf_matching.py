import numpy as np

from pytesmo.cdf_matching import (CDFMatching,
                                  _matlab_percentile_values_from_sorted,
                                  _unique_percentile_interpolation)


def cdf_matching_testdata():
    n = 1000
    src = 1 + np.arange(n, dtype=float)
    ref = src / 2
    return src, ref


def test_cdf_matching():
    src, ref = cdf_matching_testdata()

    matcher = CDFMatching().fit(src, ref)
    assert matcher.x_perc_.shape == (101,)
    assert matcher.y_perc_.shape == (101,)

    pred = matcher.predict(src)
    np.testing.assert_almost_equal(pred, ref)

    factor = pred / src
    assert np.all(np.abs(factor - 0.5) < 1e-4)


def test_cdf_matching_percentiles():
    src, ref = cdf_matching_testdata()

    percentiles = np.linspace(0, 100, 11)
    matcher = CDFMatching(percentiles=percentiles).fit(src, ref)
    assert np.all(matcher.percentiles == percentiles)
    assert matcher.x_perc_.shape == (len(percentiles),)
    assert matcher.y_perc_.shape == (len(percentiles),)

    # make sure the result is still right
    pred = matcher.predict(src)
    np.testing.assert_almost_equal(pred, ref)
    factor = pred / src
    assert np.all(np.abs(factor - 0.5) < 1e-4)


def test_cdf_matching_resize_percentiles():
    """
    Tests whether the bin resizing works.
    """
    src, ref = cdf_matching_testdata()
    src = src[0:100]
    ref = ref[0:100]

    matcher = CDFMatching(nbins=100, minobs=20).fit(src, ref)
    # there should be only 5 bins (6 percentiles)
    assert np.sum(~np.isnan(matcher.percentiles_)) == 6
    assert np.sum(~np.isnan(matcher.x_perc_)) == 6
    assert np.sum(~np.isnan(matcher.y_perc_)) == 6


def test_cdf_matching_scaling():
    # this tests whether the scaling makes the edges of the ref CDF more robust
    # against outliers
    src, ref = cdf_matching_testdata()
    ref[0] = -9

    matcher_scaled = CDFMatching(linear_edge_scaling=True).fit(src, ref)
    matcher_unscaled = CDFMatching(linear_edge_scaling=False).fit(src, ref)
    assert matcher_scaled.x_perc_.shape == (101,)
    assert matcher_scaled.y_perc_.shape == (101,)
    assert matcher_unscaled.x_perc_.shape == (101,)
    assert matcher_unscaled.y_perc_.shape == (101,)

    pred_scaled = matcher_scaled.predict(src)
    pred_unscaled = matcher_unscaled.predict(src)

    # everything above the first bin should not be affected by the scaling
    np.all(pred_scaled[20:] == pred_unscaled[20:])

    # the unscaled transformed should have -9 as lowest value
    assert pred_unscaled[0] == -9
    # the scaled transformed value should be much closer to zero
    assert -2.5 < pred_scaled[0] < 0


def test_cdf_matching_combine_invalid():
    src, ref = cdf_matching_testdata()
    ref_orig = np.array(ref)
    ref[100:200] = np.nan

    matcher = CDFMatching(combine_invalid=True).fit(src, ref)
    assert matcher.x_perc_.shape == (101,)
    assert matcher.y_perc_.shape == (101,)

    pred = matcher.predict(src)
    np.testing.assert_almost_equal(pred, ref_orig)

    factor = pred / src
    assert np.all(np.abs(factor - 0.5) < 1e-4)


def test_cdf_matching_single_bin():
    src, ref = cdf_matching_testdata()

    matcher = CDFMatching(percentiles=[0, 100]).fit(src, ref)
    assert matcher.x_perc_.shape == (2,)
    assert matcher.y_perc_.shape == (2,)

    pred = matcher.predict(src)
    np.testing.assert_almost_equal(pred, ref)

    factor = pred / src
    assert np.all(np.abs(factor - 0.5) < 1e-4)


def test_matlab_percentile_values():
    """
    Test the percentile implementation that is used in Matlab.
    """

    arr1 = np.array([1, 1, 1, 2, 2, 2, 5, 5, 6, 10, 10, 10, 10])
    percentiles = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
    perc_should = [
        1.0, 1.0, 1.0, 1.1, 2.0, 2.0, 5.0, 5.3, 8.4, 10., 10., 10., 10.
    ]
    perc = _matlab_percentile_values_from_sorted(np.sort(arr1), percentiles)
    np.testing.assert_almost_equal(perc, perc_should)


def test_unique_percentile_interpolation():
    """
    test generation of unique percentile values
    by interpolation or order k
    """

    arr1 = np.array([1, 1, 1, 2, 2, 2, 5, 5, 6, 10, 10, 10, 10])
    percentiles = np.array([0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100])
    p = _matlab_percentile_values_from_sorted(np.sort(arr1), percentiles)
    src_perc = _unique_percentile_interpolation(p, percentiles=percentiles)
    assert len(p) == len(src_perc)

    np.testing.assert_almost_equal(src_perc, [
        1., 1.025, 1.05, 1.1, 2., 3.5, 5., 5.3, 8.4, 8.93333333, 9.46666667,
        9.73333333, 10.
    ])
