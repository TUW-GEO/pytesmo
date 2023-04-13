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
        Number of bins to use for the empirical CDF. Default is 100. If
        `minobs` is set, this might be reduced in case there's not enough data
        in each bin.
    percentiles : sequence, optional
        Percentile values to use. If this is given, `nbins` is ignored. The
        percentiles might still be changed if `minobs` is given and the number
        data per bin is lower. Default is ``None``.
    linear_edge_scaling : bool, optional
        Whether to derive the edge parameters via linear regression (more
        robust, see Moesinger et al. (2020) for more info). Default is
        ``False``.
        Note that this way only the outliers in the reference (y) CDF are
        handled. Outliers in the input data (x) will not be removed and will
        still show up in the data.
    minobs : int, optional
        Minimum desired number of observations in a bin. If there is less data
        for a bin, the number of bins is reduced. Default is ``None`` (no
        resizing).
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
        The default is ``False``.

    Attributes
    ----------
    x_perc_ : np.ndarray (nbins,)
        The percentile values derived from the source (X) data. If the number
        of bins was reduced during fitting due to insufficient data, it is
        right-padded with NaNs.
    y_perc_ : np.ndarray (nbins,)
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
        minobs: int = None,
        linear_edge_scaling: bool = False,
        combine_invalid: bool = False,
    ):
        self.nbins = nbins
        self.minobs = minobs
        self.linear_edge_scaling = linear_edge_scaling
        self.combine_invalid = combine_invalid
        self.percentiles = percentiles
        if self.percentiles is not None:
            self.nbins = len(self.percentiles) - 1

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
        if isinstance(y, pd.Series):
            y = y.values
        x = _make_X_array(X)

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
        nsamples = min(len(x), len(y))
        tmp_percentiles, tmp_nbins = self._calc_percentiles(nsamples)

        if tmp_nbins == 1 and self.linear_edge_scaling:
            # for a single bin just do a linear interpolation in case linear
            # edge scaling is activated
            x_perc, y_perc = _linreg_percentiles(x_unsorted, y_unsorted)
        else:
            x_perc = _percentile_values_from_sorted(x, tmp_percentiles)
            y_perc = _percentile_values_from_sorted(y, tmp_percentiles)
            if self.linear_edge_scaling:
                y_perc = self._linear_edge_scaling(
                    x, y, x_perc, y_perc, tmp_percentiles)

        # fill self.x_perc_ and self.y_perc_ with NaN on the right in case the
        # bin size was reduced, so we always get arrays of size nbins as
        # self.x_perc_ and self.y_perc_
        self.x_perc_ = np.full(self.nbins + 1, np.nan, dtype=x.dtype)
        self.y_perc_ = np.full(self.nbins + 1, np.nan, dtype=x.dtype)
        self.percentiles_ = np.full(self.nbins + 1, np.nan, dtype=x.dtype)
        self.percentiles_[0:tmp_nbins + 1] = tmp_percentiles
        self.x_perc_[0:tmp_nbins + 1] = x_perc
        self.y_perc_[0:tmp_nbins + 1] = y_perc
        return self

    def predict(self, X):
        x = _make_X_array(X)
        isvalid = np.isfinite(x)
        prediction = np.full_like(x, np.nan)
        if not np.any(isvalid):
            return prediction
        xp = self.x_perc_[~np.isnan(self.x_perc_)]
        yp = self.y_perc_[~np.isnan(self.y_perc_)]
        if len(xp) == 0 or len(yp) == 0:
            return prediction
        else:
            try:
                spline = interpolate.InterpolatedUnivariateSpline(
                    xp, yp, k=1, ext=0)
                prediction[isvalid] = spline(x[isvalid])
                return prediction
            except ValueError:
                # happens if there are non-unique values or not enough values
                warnings.warn("Too few percentiles for chosen k.")
                return prediction

    def _calc_percentiles(self, nsamples):
        # calculate percentiles, potentially resize them
        if self.percentiles is None:
            percentiles = np.arange(
                self.nbins + 1, dtype=np.float64) / self.nbins * 100
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
        # scales the lower and upper edges of y_perc by replacing the values
        # with values inferred from a linear regression between the n
        # lowest/highest values for x and y. n is the number of data in the
        # lowest/highest y-bin, if there are more or less values in the
        # corresponding x-bin, the values will be resampled from the empirical
        # CDF. This can happen if the values are on only a few discrete levels.
        xlow = x[x <= x_perc[1]] - x_perc[1]
        ylow = y[y <= y_perc[1]] - y_perc[1]
        n = len(ylow)
        if len(xlow) != n:
            xlow = _resample_ecdf(xlow, n)
        a, _, _, _ = np.linalg.lstsq(xlow.reshape(-1, 1), ylow, rcond=None)
        y_perc[0] = y_perc[1] + a[0] * (x_perc[0] - x_perc[1])

        xhigh = x[x >= x_perc[-2]] - x_perc[-2]
        yhigh = y[y >= y_perc[-2]] - y_perc[-2]
        n = len(yhigh)
        if len(xhigh) != n:
            xhigh = _resample_ecdf(xhigh, n)
        a, _, _, _ = np.linalg.lstsq(xhigh.reshape(-1, 1), yhigh, rcond=None)
        y_perc[-1] = y_perc[-2] + a[0] * (x_perc[-1] - x_perc[-2])

        return y_perc


def _make_X_array(X):
    if isinstance(X, (pd.Series, pd.DataFrame)):
        X = X.values
    if len(X.shape) > 1:
        assert len(X.shape) == 2
        assert X.shape[1] == 1
        x = X.ravel()
    else:
        x = X
    return x


def _resample_ecdf(x_sorted, n, is_sorted=True):
    """Resample ECDF to n bins"""
    # calculate percentiles for x_sorted
    if not is_sorted:
        x_sorted = np.sort(x_sorted)
    new_percentiles = np.arange(n, dtype=float) / (n - 1) * 100.0
    return _percentile_values_from_sorted(x_sorted, new_percentiles)


def _percentile_values_from_sorted(data, percentiles):
    perc = _matlab_percentile_values_from_sorted(data, percentiles)
    return _unique_percentile_interpolation(perc, percentiles)


def _matlab_percentile_values_from_sorted(data, percentiles):
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


def _unique_percentile_interpolation(perc, percentiles):
    # make sure the percentiles are unique
    # assumes `perc` is sorted
    uniq, uniq_ind = np.unique(perc, return_index=True)
    if uniq_ind.size != len(percentiles) and uniq_ind.size > 1:
        # If there are non-unique percentile values in perc, e.g.
        # [1, 1, 1, 2, ...] corresponding to percentiles [0, 5, 10, 20, ...],
        # we will interpolate the values for 5 and 10 to be between 1 and 2.
        # uniq_ind contains the first index of the non-unique values, so
        # selecting them will return [1, 2] and [0, 20].
        new_percentiles = percentiles[uniq_ind]
        new_perc = perc[uniq_ind]
        # However, if we have non-unique indices at the end of the array, e.g.
        # [..., 8, 10, 10, 10] corresponding to percentiles [..., 85, 90, 95,
        # 100], this approach will return [8, 10] and [85, 90], meaning that
        # the last percentile values will be extrapolated.
        # To avoid this, we set the last non-unique percentile to the overall
        # last percentile
        new_percentiles[-1] = percentiles[-1]
        spline = interpolate.InterpolatedUnivariateSpline(
            new_percentiles, new_perc, k=1, ext=0)
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
