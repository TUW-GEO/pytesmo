import numpy as np

from pytesmo.cdf_matching import CDFMatching

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


def test_cdf_matching_no_scaling():
    src, ref = cdf_matching_testdata()

    matcher = CDFMatching(linear_edge_scaling=False).fit(src, ref)
    assert matcher.x_perc_.shape == (101,)
    assert matcher.y_perc_.shape == (101,)

    pred = matcher.predict(src)
    np.testing.assert_almost_equal(pred, ref)

    factor = pred / src
    assert np.all(np.abs(factor - 0.5) < 1e-4)


def test_cdf_matching_combine_invalid():
    src, ref = cdf_matching_testdata()
    ref_orig = np.array(ref)
    ref[100:200] = np.nan

    matcher = CDFMatching().fit(src, ref)
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
