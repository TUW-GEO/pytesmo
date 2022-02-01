import numpy as np
import pandas as pd
from scipy import interpolate, optimize, special
from sklearn.base import BaseEstimator, RegressorMixin
from typing import Sequence
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
    percentiles : sequence, optional
        Percentile values to use. If this is given, `nbins` is ignored. The
        percentiles might still be changed if `minobs` is given and the number
        data per bin is lower. Default is ``None``.
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
        The percentile values derived from the reference (y) data. If the
        number of bins was reduced during fitting due to insufficient data, it
        is right-padded with NaNs.

    Notes
    -----
    This implementation does not do any temporal matching of the reference and
    source datasets. If this is required, this has to be done beforehand.
    """

    def __init__(
        self,
        nbins: int = 100,
        percentiles: Sequence = None,
        minobs: int = 20,
        linear_edge_scaling: bool = True,
        combine_invalid: bool = True,
    ):
        self.nbins = nbins
        self.minobs = minobs
        self.linear_edge_scaling = linear_edge_scaling
        self.combine_invalid = combine_invalid
        self.percentiles = percentiles

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

        if tmp_nbins == 1 and self.linear_edge_scaling:
            # for a single bin just do a linear interpolation in case linear
            # edge scaling is activated
            x_perc, y_perc = _linreg_percentiles(x_unsorted, y_unsorted)
        else:
            x_perc = _matlab_percentile_from_sorted(x, tmp_percentiles)
            y_perc = _matlab_percentile_from_sorted(y, tmp_percentiles)
            if self.linear_edge_scaling:
                x_perc, y_perc = self._linear_edge_scaling(
                    x, y, x_perc, y_perc, tmp_percentiles)

        # fill self.x_perc_ and self.y_perc_ with NaN on the right in case the
        # bin size was reduced, so we always get arrays of size nbins as
        # self.x_perc_ and self.y_perc_
        self.x_perc_ = np.zeros(self.nbins, dtype=x.dtype) * np.nan
        self.y_perc_ = np.zeros(self.nbins, dtype=x.dtype) * np.nan
        self.percentiles_ = np.zeros(self.nbins, dtype=x.dtype) * np.nan
        self.percentiles_[0:tmp_nbins + 1] = tmp_percentiles
        self.x_perc_[0:tmp_nbins + 1] = x_perc
        self.y_perc_[0:tmp_nbins + 1] = y_perc
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
            try:
                spline = interpolate.InterpolatedUnivariateSpline(
                    xp, yp, k=1, ext=0)
                return spline(x)
            except ValueError:
                # happens if there are non-unique values or not enough values
                warnings.warn("Too few percentiles for chosen k.")
                return np.full_like(x, np.nan)

    def _calc_percentiles(self, nsamples):
        # calculate percentiles, potentially resize them
        if self.percentiles is None:
            percentiles = np.arange(
                self.nbins, dtype=np.float64) / self.nbins * 100
        else:
            percentiles = self.percentiles

        minbinsize = np.min(np.diff(percentiles))
        if (self.minobs is not None and
                nsamples * minbinsize / 100 < self.minobs):
            warnings.warn("The bins have been resized")
            nbins = np.int32(np.floor(nsamples / self.minobs))
            if nbins == 0:
                nbins = 1
            elif nbins > len(percentiles) - 1:
                nbins = len(percentiles) - 1

            percentiles = np.arange(nbins + 1, dtype=np.float64) / nbins * 100
        else:
            nbins = len(percentiles) - 1
        return percentiles, nbins

    def _linear_edge_scaling(self, x, y, x_perc, y_perc, percentiles):
        # the data are sorted, so the first n values are the values
        # corresponding to y <= y_perc[1]
        n = np.sum(y <= y_perc[1])
        xlow = x[:n] - x_perc[1]
        ylow = y[:n] - y_perc[1]
        a, _, _, _ = np.linalg.lstsq(xlow.reshape(-1, 1), ylow, rcond=None)
        y_perc[0] = y_perc[1] + a[0] * (x_perc[0] - x_perc[1])

        # the last n values correspond to the values in the last bin
        n = np.sum(y >= y_perc[-2])
        xhigh = x[-n:] - x_perc[1]
        yhigh = y[-n:] - y_perc[1]
        a, _, _, _ = np.linalg.lstsq(xhigh.reshape(-1, 1), yhigh, rcond=None)
        y_perc[-1] = y_perc[-2] + a[0] * (x_perc[-1] - x_perc[-2])

        return x_perc, y_perc


def _resample_ecdf(x_sorted, n):
    """Resample ECDF to n bins"""
    # calculate percentiles for x_sorted
    new_percentiles = np.arange(n, dtype=float) / (n - 1) * 100.0
    return _matlab_percentile_from_sorted(x_sorted, new_percentiles)


def _matlab_percentile_from_sorted(data, percentiles):
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

    # make sure the percentiles are unique
    uniq, uniq_ind = np.unique(perc, return_index=True)
    if uniq_ind.size != len(percentiles) and uniq_ind.size > 1:
        spline = interpolate.InterpolatedUnivariateSpline(
            percentiles[uniq_ind], perc[uniq_ind], k=1, ext=0)
        perc = spline(percentiles)
    return perc


def _linreg_percentiles(x, y):
    """Calculate percentiles via linear regression."""
    A = np.reshape(x - np.mean(x), (-1, 1))
    b = y - np.mean(y)
    slope, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    intercept = np.mean(y) - np.mean(x) * slope[0]
    x_perc = [0, 1]
    y_perc = [intercept, intercept + slope[0]]
    return x_perc, y_perc
