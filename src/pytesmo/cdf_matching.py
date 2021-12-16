import numpy as np
import pandas as pd
from scipy import interpolate, optimize, special
from sklearn.base import BaseEstimator, RegressorMixin
import warnings

from typing import Union


class CDFMatching(RegressorMixin, BaseEstimator):
    """
    Predicts variable from single other variable by CDF matching.

    Parameters
    ----------
    nbins : int, optional
        Number of bins to use for the empirical CDF. Default is 100. This might
        be reduced in case there's not enough data in each bin.
    lin_edge_scaling : bool, optional
        Whether to derive the edge parameters via linear regression (more
        robust, see Moesinger et al. (2020) for more info). Default is
        ``True``.
    minobs : int, optional
        Minimum desired number of observations in a bin. If there is less data
        for a bin, the number of bins is reduced. Default is 20. Set to
        ``None`` if no bin resizing should be performed.
    combine_invalid : bool, optional
        Optional feature to combine the masks of invalid data (NaN, Inf) of
        both source (X) and reference (y) data passed to `fit`. This only makes
        sense if X and y are both timeseries data corresponding to the same
        index. In this case, this makes sures that data is only used if values
        for X and y are available, so that seasonal patterns in missing values
        in one of them do not lead to distortions. (For example, if X is
        available the whole year, but y is only available during summer, the
        distribution of y should not be matched against the whole year CDF of
        X, because that could introduce systematic seasonal biases).

    Attributes
    ----------
    src_perc_ : np.ndarray (nbins,)
        The percentile values derived from the source (X) data. If the number
        of bins was reduced during fitting due to insufficient data, it is
        right-padded with NaNs.
    ref_perc_ : np.ndarray (nbins,)
        The percentile values derived from the reference (y) data. If the number
        of bins was reduced during fitting due to insufficient data, it is
        right-padded with NaNs.

    Notes
    -----
    This implementation does not do any temporal matching of the reference and
    source datasets. If this is required, this has to be done beforehand.
    """

    def __init__(
        self,
        nbins: int = 100,
        minobs: int = 20,
        linear_edge_scaling: bool = True,
        combine_invalid: bool = True,
    ):
        self.nbins = nbins
        self.minobs = minobs
        self.linear_edge_scaling = linear_edge_scaling
        self.combine_invalid = combine_invalid

    def fit(
        self,
        X: Union[np.ndarray, pd.Series, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
    ):
        """
        Derive the CDF matching parameters.

        Parameters
        ----------
        X : array_like
           An array/pd.Series or a matrix/pd.DataFrame with a single column.
        y : array_like
           An array/pd.Series of reference data.
        """

        # make sure that x and y are 1D numpy arrays
        if isinstance(X, (pd.Series, pd.DataFrame)):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        if len(X.shape) > 1:
            assert len(X.shape) == 2
            assert X.shape[1] == 1
            x = X.ravel()
        else:
            x = X

        # drop invalid values and pre-sort to avoid multiple resorting later
        isvalid_x = np.isfinite(x)
        isvalid_y = np.isfinite(y)
        if len(x) == len(y) and self.combine_invalid:
            isvalid_x = isvalid_x & isvalid_y
            isvalid_y = isvalid_x
        x_unsorted = x[isvalid_x]
        y_unsorted = y[isvalid_y]
        x = np.sort(x_unsorted)
        y = np.sort(y_unsorted)

        # calculate percentiles, potentially resize them
        tmp_percentiles, tmp_nbins = self._calc_percentiles(len(x))

        # resample if necessary to have the same amount of data for source and
        # ref
        if len(x) != len(y):
            max_samples = max(len(x), len(y))
            if len(x) < len(y):
                x = _resample_ecdf(x, max_samples)
            else:
                y = _resample_ecdf(y, max_samples)

        # for a single bin just do a linear interpolation
        if tmp_nbins == 1 and self.linear_edge_scaling:
            x_perc, y_perc = _linreg_percentiles(x_unsorted, y_unsorted)
        else:

            x_perc = _matlab_percentile(x, tmp_percentiles)
            y_perc = _matlab_percentile(y, tmp_percentiles)
            x_all_equal = x[0] == x[-1]  # because it's sorted
            if not x_all_equal:
                x_perc, y_perc = self._calc_unique_percentiles(
                    x_perc, y_perc, tmp_percentiles
                )
            if self.linear_edge_scaling:
                x_perc, y_perc = self._linear_edge_scaling(
                    x, y, x_perc, y_perc, tmp_percentiles
                )

        # fill self.x_perc_ and self.y_perc_ with NaN on the right in case the
        # bin size was reduced, so we always get arrays of size nbins as
        # self.x_perc_ and self.y_perc_
        self.x_perc_ = np.zeros(self.nbins + 1, dtype=x.dtype) * np.nan
        self.y_perc_ = np.zeros(self.nbins + 1, dtype=x.dtype) * np.nan
        self.percentiles_ = np.zeros(self.nbins + 1, dtype=x.dtype) * np.nan
        self.percentiles_[0 : tmp_nbins + 1] = tmp_percentiles
        self.x_perc_[0 : tmp_nbins + 1] = x_perc
        self.y_perc_[0 : tmp_nbins + 1] = y_perc
        return self

    def predict(self, X):
        if isinstance(X, (pd.Series, pd.DataFrame)):
            X = X.values
        if len(X.shape) > 1:
            assert len(X.shape) == 2
            assert X.shape[1] == 1
            x = X.ravel()
        else:
            x = X
        xp = self.x_perc_[~np.isnan(self.x_perc_)]
        yp = self.y_perc_[~np.isnan(self.y_perc_)]
        if len(xp) == 0 or len(yp) == 0:
            return np.zeros(len(x), dtype=x.dtype) * np.nan
        else:
            spline = interpolate.InterpolatedUnivariateSpline(
                xp, yp, k=1, ext=0
            )
            return spline(x)

    def _calc_percentiles(self, nsamples):
        # calculate percentiles, potentially resize them
        percentiles = (
            np.arange(self.nbins + 1, dtype=np.float64) / self.nbins * 100
        )
        minbinsize = percentiles[1] - percentiles[0]
        if (
            self.minobs is not None
            and nsamples * minbinsize / 100 < self.minobs
        ):
            warnings.warn("The bins have been resized")
            nbins = np.int32(np.floor(nsamples / self.minobs))
            if nbins == 0:
                nbins = 1
            elif nbins > len(percentiles) - 1:
                nbins = len(percentiles) - 1

            percentiles = np.arange(nbins + 1, dtype=np.float64) / nbins * 100
        else:
            nbins = self.nbins
        return percentiles, nbins

    def _calc_unique_percentiles(self, x_perc, y_perc, percentiles):
        x_perc = _unique_percentiles_beta(x_perc, percentiles)

        # make sure ref percentiles are unique if not then linear interpolation
        # is used
        # TODO: do this without using np.unique, we already know the data is
        # sorted, so it's much easier to do this by hand
        uniq, uniq_ind = np.unique(
            y_perc, return_index=True
        )
        if uniq_ind.size != len(percentiles) and uniq_ind.size > 1:
            spline = interpolate.InterpolatedUnivariateSpline(
                percentiles[uniq_ind], y_perc[uniq_ind], k=1, ext=0
            )
            y_perc = spline(percentiles)
            spline = interpolate.InterpolatedUnivariateSpline(
                percentiles[uniq_ind], x_perc[uniq_ind], k=1, ext=0
            )
            x_perc = spline(percentiles)
        return x_perc, y_perc

    def _linear_edge_scaling(self, x, y, x_perc, y_perc, percentiles):
        xlow = x[x <= x_perc[1]] - x_perc[1]
        ylow = y[y <= y_perc[1]] - y_perc[1]
        n = min(len(xlow), len(ylow))
        xlow, ylow = (
            xlow[0:n],
            ylow[0:n],
        )  # avoids issues due to rouding errors
        a, _, _, _ = np.linalg.lstsq(xlow.reshape(-1, 1), ylow, rcond=None)
        y_perc[0] = y_perc[1] + a[0] * (x_perc[0] - x_perc[1])

        xhigh = x[x >= x_perc[-2]] - x_perc[-2]
        yhigh = y[y >= y_perc[-2]] - y_perc[-2]
        n = min(len(xhigh), len(yhigh))
        xhigh, yhigh = (
            xhigh[-n:],
            yhigh[-n:],
        )  # avoids issues due to rouding errors
        a, _, _, _ = np.linalg.lstsq(xhigh.reshape(-1, 1), yhigh, rcond=None)
        y_perc[-1] = y_perc[-2] + a[0] * (x_perc[-1] - x_perc[-2])

        return x_perc, y_perc


def _resample_ecdf(x_sorted, n):
    # get percentiles between 0 and 1
    # p = (stats.rankdata(x_sorted) - 1) / (len(x_sorted)-1)
    # this is a bit faster, but doesn't handle ties
    p = np.arange(len(x_sorted) + 1, dtype=float) / len(x_sorted)
    new_p = np.arange(n + 1, dtype=float) / n
    return np.interp(new_p, p, x_sorted)


def _matlab_percentile(data, percentiles):
    """
    Calculate percentiles in the way Matlab and IDL do it.

    By using interpolation between the lowest an highest rank and the
    minimum and maximum outside.

    Parameters
    ----------
    data: numpy.ndarray
        sorted input data
    percentiles: numpy.ndarray
        percentiles at which to calculate the values
    Returns
    -------
    perc: numpy.ndarray
        values of the percentiles
    """
    p_rank = 100.0 * (np.arange(data.size) + 0.5) / data.size
    perc = np.interp(percentiles, p_rank, data, left=data[0], right=data[-1])
    return perc


def _unique_percentiles_beta(perc_values, percentiles):
    """
    Compute unique percentile values by fitting the CDF of a beta distribution
    to the percentiles.

    Parameters
    ----------
    perc_values: list or numpy.ndarray
        calculated values for the given percentiles
    percentiles: list or numpy.ndarray
        Percentiles to use for CDF matching

    Returns
    -------
    uniq_perc_values: numpy.ndarray
        Unique percentile values generated through fitting
        the CDF of a beta distribution.

    Raises
    ------
    RuntimeError
        If no fit could be found.
    """

    # normalize between 0 and 1
    # since the data is sorted, non-unique values show up as zeros in diff
    is_unique = np.all(np.diff(perc_values) != 0)
    if not is_unique:
        min_value = np.min(perc_values)
        perc_values = perc_values - min_value
        max_value = np.max(perc_values)
        perc_values = perc_values / max_value
        percentiles = np.asanyarray(percentiles)
        percentiles = percentiles / 100.0

        p, ier = optimize.curve_fit(
            lambda x, a, b: special.betainc(a, b, x), percentiles, perc_values
        )
        uniq_perc_values = special.betainc(p[0], p[1], percentiles)
        uniq_perc_values = uniq_perc_values * max_value + min_value
    else:
        uniq_perc_values = perc_values
    return uniq_perc_values


def _linreg_percentiles(x, y):
    """Calculate percentiles via linear regression."""
    A = np.reshape(x - np.mean(x), (-1, 1))
    b = y - np.mean(y)
    slope, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    intercept = np.mean(y) - np.mean(x) * slope[0]
    x_perc = [0, 1]
    y_perc = [intercept, intercept + slope[0]]
    return x_perc, y_perc
