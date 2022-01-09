import numpy as np

from pytesmo.cdf_matching import CDFMatching


def test_cdf_matching():
    n = 1000
    src = 1 + np.arange(n, dtype=float)
    ref = src / 2

    matcher = CDFMatching().fit(src, ref)
    assert matcher.x_perc_.shape == (100,)
    assert matcher.y_perc_.shape == (100,)

    pred = matcher.predict(src)
    np.testing.assert_almost_equal(pred, ref)

    factor = pred / src
    assert np.all(np.abs(factor - 0.5) < 1e-4)
