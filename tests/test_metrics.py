from datetime import datetime
import numpy as np
import numpy.testing as nptest
import pandas as pd
import pytest

import pytesmo.metrics
from pytesmo.metrics import *
import pytesmo.metrics.deprecated as deprecated


@pytest.fixture
def testdata():
    # the seed avoids random failure
    np.random.seed(0)

    # generate random data that is correlated with r = 0.8
    cov = np.array([[1, 0.8], [0.8, 1]])
    X = np.linalg.cholesky(cov) @ np.random.randn(2, 1000)
    x, y = X[0, :], X[1, :]
    y = 1.1 * y + 0.5
    return x, y


@pytest.fixture
def arange_testdata():
    x = np.arange(10, dtype=float)
    y = np.arange(10, dtype=float) + 2
    return x, y


has_ci = [
    "bias",
    "msd",
    "rmsd",
    "nrmsd",
    "ubrmsd",
    "mse_bias",
    "pearson_r",
    "spearman_r",
    "kendall_tau",
]

no_ci = [
    "aad",
    "mad",
    "mse_corr",
    "mse_var",
    "nash_sutcliffe",
    "index_of_agreement",
]


def test_analytical_ci_availability(testdata):
    for funcname in has_ci:
        func = getattr(pytesmo.metrics, funcname)
        assert has_analytical_ci(func)

    for funcname in no_ci:
        func = getattr(pytesmo.metrics, funcname)
        assert not has_analytical_ci(func)


def test_analytical_cis(testdata):
    x, y = testdata
    for funcname in has_ci:
        func = getattr(pytesmo.metrics, funcname)
        m, lb, ub = with_analytical_ci(func, x, y)
        m10, lb10, ub10 = with_analytical_ci(func, x, y, alpha=0.1)
        assert m == m10
        assert lb < m
        assert m < ub
        assert lb < lb10
        assert ub > ub10


@pytest.mark.slow
def test_bootstrapped_cis(testdata):
    x, y = testdata
    for funcname in has_ci:
        func = getattr(pytesmo.metrics, funcname)
        m, lb, ub = with_analytical_ci(func, x, y, alpha=0.1)
        m_bs, lb_bs, ub_bs = with_bootstrapped_ci(
            func, x, y, alpha=0.1, nsamples=1000
        )
        assert m == m_bs
        assert lb_bs < ub_bs
        if funcname != "nrmsd":
            # nrmsd is a bit unstable when bootstrapping, due to the data
            # dependent normalization that is applied
            assert abs(ub - ub_bs) < 1e-2
            assert abs(lb - lb_bs) < 1e-2
        else:
            assert abs(ub - ub_bs) < 1e-1
            assert abs(lb - lb_bs) < 1e-1
    for funcname in no_ci:
        m, lb, ub = with_bootstrapped_ci(func, x, y, alpha=0.1, nsamples=1000)
        assert lb < m
        assert m < ub


# expected values of the metrics, to test whether it works
def test_expected_values(testdata):
    x, y = testdata

    expected_msd_bias = 0.5 ** 2
    expected_msd_var = 0.1 ** 2
    expected_msd_corr = 2 * 1 * 1.1 * (1 - 0.8)
    expected_msd = expected_msd_bias + expected_msd_corr + expected_msd_var
    expected_values = {
        "bias": -0.5,
        "msd": expected_msd,
        "rmsd": np.sqrt(expected_msd),  # actually not totally true
        "ubrmsd": np.sqrt(expected_msd_corr + expected_msd_var),  # not quite
        "mse_corr": expected_msd_corr,
        "mse_bias": expected_msd_bias,
        "mse_var": expected_msd_var,
        "pearson_r": 0.8,
    }

    for metric in expected_values:
        func = getattr(pytesmo.metrics, metric)
        m, lb, ub = with_bootstrapped_ci(func, x, y)
        e = expected_values[metric]
        assert lb < e
        assert e < ub


def test_tcol_metrics():
    n = 1000000

    mean_signal = 0.3
    sig_signal = 0.2
    signal = np.random.normal(mean_signal, sig_signal, n)

    sig_err_x = 0.02
    sig_err_y = 0.07
    sig_err_z = 0.04
    err_x = np.random.normal(0, sig_err_x, n)
    err_y = np.random.normal(0, sig_err_y, n)
    err_z = np.random.normal(0, sig_err_z, n)

    alpha_y = 0.2
    alpha_z = 0.5

    beta_y = 0.9
    beta_z = 1.6

    x = signal + err_x
    y = alpha_y + beta_y * (signal + err_y)
    z = alpha_z + beta_z * (signal + err_z)

    beta_pred = 1.0 / np.array((1, beta_y, beta_z))
    err_pred = np.array((sig_err_x, sig_err_y, sig_err_z))
    snr_pred = np.array(
        (
            (sig_signal / sig_err_x),
            (sig_signal / sig_err_y),
            (sig_signal / sig_err_z),
        )
    )

    snr, err, beta = tcol_metrics(x, y, z, ref_ind=0)

    nptest.assert_almost_equal(beta, beta_pred, decimal=2)
    nptest.assert_almost_equal(err, err_pred, decimal=2)
    nptest.assert_almost_equal(
        np.sqrt(10 ** (snr / 10.0)), snr_pred, decimal=1
    )


def test_bias(arange_testdata):
    """
    Test for bias
    """
    # example 1
    x, y = arange_testdata

    b_pred = -2
    b_obs = bias(x, y)

    nptest.assert_equal(b_obs, b_pred)

    # example 2
    x = np.arange(10) * 1.0
    y = np.arange(20, 30) * 1.0

    b_pred = 20.
    b_obs = bias(y, x)

    nptest.assert_equal(b_obs, b_pred)


def test_aad(arange_testdata):
    """
    Test for average absolute deviation
    """
    # example 1
    x, y = arange_testdata
    dev_pred = 2.
    dev_obs = aad(x, y)

    nptest.assert_equal(dev_obs, dev_pred)

    # example 2, with outlier
    y[-1] = 201.
    dev_pred = 21.
    dev_obs = aad(x, y)

    nptest.assert_equal(dev_obs, dev_pred)


def test_mad(arange_testdata):
    """
    Test for median absolute deviation
    """
    # example 1
    x, y = arange_testdata
    dev_pred = 2.
    dev_obs = mad(x, y)

    nptest.assert_equal(dev_obs, dev_pred)

    # example 2, with outlier
    y[-1] = 201.
    dev_pred = 2.
    dev_obs = mad(x, y)

    nptest.assert_equal(dev_obs, dev_pred)


def test_rmsd(arange_testdata):
    """
    Test for rmsd
    """
    # example 1
    x, y = arange_testdata

    rmsd_pred = 2.
    rmsd_obs = rmsd(x, y)

    nptest.assert_equal(rmsd_obs, rmsd_pred)

    # example 2, with outlier
    y[-1] = 100.

    rmsd_pred = np.sqrt(831.7)
    rmsd_obs = rmsd(x, y)

    nptest.assert_almost_equal(rmsd_obs, rmsd_pred, 6)


def test_ubrmsd(arange_testdata):
    """
    Test for ubrmsd
    """
    # example 1
    x, y = arange_testdata

    ubrmsd_pred = 0
    ubrmsd_obs = ubrmsd(x, y)

    nptest.assert_equal(ubrmsd_obs, ubrmsd_pred)
    # aslo check consistency with direct formula
    ubrmsd_direct = np.sqrt(rmsd(x, y) ** 2 - bias(x, y)**2)
    nptest.assert_equal(ubrmsd_obs, ubrmsd_direct)

    # example 2, with outlier
    y[-1] = 100.

    ubrmsd_pred = 26.7
    ubrmsd_obs = ubrmsd(x, y)

    nptest.assert_almost_equal(ubrmsd_obs, ubrmsd_pred, 6)
    # aslo check consistency with direct formula
    ubrmsd_direct = np.sqrt(rmsd(x, y) ** 2 - bias(x, y)**2)
    nptest.assert_almost_equal(ubrmsd_obs, ubrmsd_direct)


def test_msd(arange_testdata):
    """
    Test for msd
    """
    # example 1
    x, y = arange_testdata

    mse_pred = 4.
    mse_bias_pred = 2. ** 2
    mse_obs = msd(x, y)
    mse_bias_obs = mse_bias(x, y)

    nptest.assert_equal(mse_obs, mse_pred)
    nptest.assert_equal(mse_bias_obs, mse_bias_pred)

    # example 2, with outlier
    y[-1] = 51.

    mse_pred = 180.
    mse_bias_pred = 36.
    mse_obs = msd(x, y)
    mse_bias_obs = mse_bias(x, y)

    nptest.assert_almost_equal(mse_obs, mse_pred, 6)
    nptest.assert_almost_equal(mse_bias_obs, mse_bias_pred, 6)


def test_mse_decomposition(arange_testdata):
    # example 1
    x, y = arange_testdata

    mse_pred = 4.
    mse_bias_pred = 2. ** 2
    mse_obs, _, mse_bias, _ = mse_decomposition(x, y)

    nptest.assert_equal(mse_obs, mse_pred)
    nptest.assert_equal(mse_bias, mse_bias_pred)

    # example 2, with outlier
    y[-1] = 51.

    mse_pred = 180.
    mse_bias_pred = 36.
    mse_obs, _, mse_bias, _ = mse_decomposition(x, y)

    nptest.assert_almost_equal(mse_obs, mse_pred, 6)
    nptest.assert_almost_equal(mse_bias, mse_bias_pred, 6)


def test_rmsd_mse():
    """
    Test for rmsd and mse
    """
    # example 1
    x = np.random.randn(1000)
    y = np.random.randn(1000)

    rmsd_pred = rmsd(x, y)
    mse_pred, _, _, _ = mse_decomposition(x, y)

    nptest.assert_almost_equal(rmsd_pred ** 2, mse_pred, 6)


def test_rolling_pr_rmsd():
    # setup test data
    window_size = '30d'
    min_periods = 2
    center = True

    startdate = datetime(2000, 1, 1)
    enddate = datetime(2000, 12, 31)
    dt_index = pd.date_range(start=startdate, end=enddate, freq='D')

    names = ['ref', 'k1', 'k2', 'k3']
    # always 0.5
    np.random.seed(10)
    df = pd.DataFrame(
        index=dt_index,
        data={name: 0.5+0.01*np.random.randn(dt_index.size) for name in names}
    )

    df['k1'] += 0.2  # some positive bias
    df['k2'] -= 0.2  # some negative bias
    df['k3'] -= 0.3  # some more negative bias

    data = df
    xy = data.to_numpy()
    timestamps = data.index.to_julian_date().values
    window_size_jd = (
        pd.Timedelta(window_size).to_numpy()/np.timedelta64(1, 'D')
    )

    pr, rmsd = deprecated.rolling_pr_rmsd(
        timestamps, xy, window_size_jd, center, min_periods
    )

    prnew, rmsdnew = rolling_pr_rmsd(
        timestamps, xy[:, 0], xy[:, 1], window_size_jd, center, min_periods
    )

    nptest.assert_almost_equal(pr[:, 0], prnew[:, 0])
    # p-values might not match up to machine precision, especially if they are
    # close to 1
    nptest.assert_almost_equal(pr[:, 1], prnew[:, 1], 4)
    nptest.assert_almost_equal(rmsd, rmsdnew)
    
