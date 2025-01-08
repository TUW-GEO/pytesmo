"""
The following python implementation of the DCT-PLS algorithm (Garcia 2010)
for python 3 and is based on:

- Original MATLAB implementation: https://www.biomecardio.com/matlab/smoothn.m
- Python 2 implementation: https://github.com/profLewis/geogg122

References:
-----------
Garcia, D. (2010) 'Robust smoothing of gridded data in one and higher
dimensions with missing values',
Computational Statistics & Data Analysis, 54(4), pp. 1167–1178.
Available at: https://doi.org/10.1016/j.csda.2009.09.020

TODO:
 - [] : average leverage for h from matlab?
 - [] : add field for input uncertainties? Propagate through interpolation
"""

import sys
import warnings
import scipy.optimize as opt
import numpy as np
from typing import Literal, Optional, Union, Tuple, Annotated
from dataclasses import dataclass
from scipy.fftpack import dct, idct
from scipy.ndimage import distance_transform_edt
from collections import OrderedDict
from collections.abc import Iterable
import logging

from pytesmo.utils import pytesmolog, loglevel


@dataclass
class ValueRange:
    min: float
    max: float


logger = pytesmolog
streamHandler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
streamHandler.setFormatter(formatter)


def calc_init_guess(
        y: np.ndarray,
        mask: Optional[np.ndarray] = None,
        coeff: Optional[float] = 0.1,
        sampling: Optional[Union[float, Iterable[float]]] = None,
        return_distances: Optional[bool] = True
) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
    """
    Initial smoothing guess. This also fills any gaps in the data using a
    a nearest neighbour search. Later on, the initial results will be improved
    using the DCT-PLS. The sampling parameter allows a different unit
    for each dimension.

    Parameters
    ----------
    y: np.ndarray
        Data (with gaps) to interpolate
    mask: np.ndarray, optional (default: None)
        Boolean array indicating missing values
    coeff: float, optional (default: 0.1)
        Percent relative (0-1) of DCT coefficients to use to generate
        the initial guess. By default, the first 10% are used.
    sampling : float | Iterable[float] | None, optional (default: None)
        Spacing of elements along each dimension. If a sequence, must be of
        length equal to the input rank; if a single number, this is used for
        all axes. If not specified, a grid spacing of unity is implied.
        ------
        The default sampling does not prefer any axis (i.e., 1 for all).
        We can e.g. pass (0.5, 1, 1) for 3D data, to prefer the
        temporal neighbours over spatial neighbours (assuming that dim=0 refers
        to different time stamps).
    return_distances: bool, optional (default: True)
        Return Euclidean distance to the nearest valid data point in addition
        to initial interpolation and smoothing result.

    Returns
    -------
    z: np.ndarray
        Initial guess of smoothened and interpolated data
    dist: np.ndarray or None
        Euclidean distance to the nearest observation (only if mask is passed).
        Only if `return_distances=True` (otherwise this is None)
    """

    z = y.copy()

    dist = None
    if mask is not None:
        if mask.shape != z.shape:
            raise ValueError("Data and missing values have different shapes.")
        if np.any(mask):
            # For the initial guess, fill gaps using a NN interpolation
            ret = distance_transform_edt(
                mask.astype(int),
                return_distances=return_distances,
                return_indices=True,
                sampling=np.full(z.ndim, 1) if sampling is None else sampling)
            if return_distances is True:
                dist, nn_inds = ret
                dist = dist.astype(np.float32)
            else:
                nn_inds = ret
            z[mask] = z[tuple(ind[mask] for ind in nn_inds)]

    # coarse fast initial smoothing
    z = dctNd(z, inverse=False)
    z_coarse = np.full(z.shape, 0.)
    if coeff > 1 or coeff <= 0:
        raise ValueError("Choose `coeff` as 0 < coeff <= 1")
    shape_reduced = np.ceil(np.array(z.shape) * coeff).astype(int)
    loc = tuple([slice(None, s) for s in shape_reduced])
    z_coarse[loc] = z[loc]
    z = dctNd(z_coarse, inverse=True)

    return z.astype(y.dtype), dist


def dctNd(data: np.ndarray,
          type: Optional[int] = 2,
          norm: Optional[Union[str, None]] = 'ortho',
          inverse: Optional[bool] = False) -> np.ndarray:
    """
    Applies discrete cosine transform function to up to 3 dimensions of data.
    Nans in data are ignored

    Parameters:
    ----------
    data: np.ndarray
        1, 2, or 3-dimensional data to calculate DCT for.
    type: int, optional (default: 2)
        DCT Type to use (see scipy DCT docs)
    norm: str or None, optional (default: 'ortho')
        Normalization mode for DCT (see scipy DCT docs)
    inverse: bool, optional (default: False)
        Use the inverse function.
    """
    func = idct if inverse else dct

    nd = len(data.shape)
    kwargs = dict(norm=norm, type=type)

    if nd == 1:
        dctNd = func(data, axis=0, **kwargs)
    elif nd == 2:
        dctNd = func(func(data, **kwargs, axis=0), **kwargs, axis=1)
    elif nd == 3:
        dctNd = func(
            func(func(data, axis=0, **kwargs), axis=1, **kwargs),
            axis=2,
            **kwargs)
    else:
        raise ValueError(
            f"Expected 1, 2, or 3 dimensional dat, but got {nd} dims.")

    return dctNd


def gcv(p: float,
        Lambda: np.ndarray,
        DCTy: np.ndarray,
        y: np.ndarray,
        smoothOrder: Optional[int] = 2,
        Wtot: Optional[np.ndarray] = None,
        score_only: Optional[bool] = False):
    """
    Find the best smoothing parameter. I.e. the parameter that minimizes the
    generalised cross validation (GCV) score.

    ===================================================
    smooth = argmin(GCV)
    GCV(s) = (wRSS / (n - n_{miss})) / (1 - Tr(H) / n)²
    ===================================================

    Parameters:
    ----------
    p: float
        Defines the order of smoothing for parameter as 10 ** p
    Lambda: np.ndarray
        Diagonal Eigenvalue matrix of D defined by Yueh (2005)
        with: \\lambda_i = -2 + 2*cos( (i-1) * \\pi / n)
    DCTy: np.ndarray
        Output array of dctNd
    y: np.ndarray
        Input data array
    smoothOrder: int, optional (default: 2)
        Exponential of lambda used for smoothing.
        Either 1 or 2
    Wtot: np.ndarray, optional (default: None)
        Weights for each y value to use (same shape as y).
        Elements in Wtot for which the counterpart in y is NaN are ignored.
        When None are passed, then the unweighted implementation, which
        is much faster, will be used.
    score_only: bool, optional (default: False)
        Only return the score for p, not Gamma and TrH (for optimisation)
        This is required for the bounded minimization which requires a single
        return value.

    Returns:
    --------
    score: np.ndarray or None
        GCV score. 0, 1 or 2 dimensional ... dim=Lambda.ndim-1
    smooth: float or None
        10**p, the smoothing parameter
    Gamma: np.ndarray or None
        Gamma when applying s
    TrH: np.ndarray or None
        Track of Hat when applying s
    """
    if smoothOrder not in [1, 2]:
        raise ValueError(f"SmoothOrder must be either 1 or 2 but "
                         f"{smoothOrder} was passed.")
    smooth = 10 ** p

    # todo: Use Eq 12 for large n?
    Gamma = (1. / (1 + smooth * Lambda ** smoothOrder))

    mask_finite = np.array(np.isfinite(y)).astype(bool)

    nfin = np.count_nonzero(mask_finite)
    nmiss = mask_finite.size - nfin

    # Residual sum-of-squares calculation
    if Wtot is None:
        # very much faster: does not require any inverse DCT
        RSS = np.linalg.norm(DCTy * (Gamma - 1.)) ** 2
    else:
        # take account of the weights to calculate RSS:
        yhat = dctNd(Gamma * DCTy, inverse=True)
        RSS = np.linalg.norm(
            np.sqrt(Wtot[mask_finite]) *
            (y[mask_finite] - yhat[mask_finite])) ** 2
        if not (yhat.ndim == y.ndim):
            raise ValueError("`yhat` and `y` must have the same shape")

    TrH = np.sum(Gamma)
    score = RSS / float(nfin) / (1. - TrH / float(nmiss + nfin)) ** 2
    score = score.astype(np.float32)

    if score_only:
        return score.flatten()
    else:
        return score.flatten(), smooth, Gamma, TrH


def sumnd(x):
    """
    Apply numpy sum over all available dimensions

    Parameters
    ----------
    x: np.ndarray
        1, 2 or 3-dimensional data array

    Returns
    -------
    sum: float
        Sum over all available dimensions

    """
    if x.ndim == 1:
        return x
    elif x.ndim == 2:
        return np.sum(x, axis=1)
    elif x.ndim == 3:
        return np.sum(x, axis=1)
    else:
        raise NotImplementedError("sumnd is only available for 1,2,3 dim data")


def RobustWeights(residuals,
                  h,
                  mask=None,
                  method: Literal["cauchy", "talworth", "bisquare"] = 'cauchy',
                  simple=True):
    """
    Compute weights from residuals.

    Parameters
    ----------
    residuals: np.ndarray
        Residuals between previous and current smoothing run. Larger residuals
        with result in larger weights. NaNs will get weight 0.
    h: float
        todo
    mask: np.ndarray, optional (default: None)
        Boolean array of same shape as residuals.
        True indicates residuals to ignore when creating new weights.
        By default, all residuals are used.
    method: str, optional (default: 'cauchy')
        A method to compute the weights.
        One of: "cauchy", "talworth", "bisquare"
        todo: Note that at the moment, only "cauchy" is implemented.
    simple: bool, optional (default: True)
        Use numpy absolute to compute the residuals.

    Returns
    -------
    weights: np.ndarray
        Weights based on passed residuals and the chosen method.
        Nans have weight 0.

    """
    _methods = ['cauchy', 'talworth', 'bisquare']

    def sabs(x):
        return np.sqrt(sumnd(np.abs(x) ** 2))

    if simple:
        f = np.abs
    else:
        f = sabs

    if mask is None:
        mask = np.full(residuals.shape, True)

    AD = sabs(residuals[~mask] - np.median(residuals[~mask]))

    if residuals.ndim == 3:
        res = residuals.flatten()
        MAD = np.median(AD)  # median absolute deviation
        u = f(res / (1.4826 * MAD)) / np.sqrt(1 - h)  # studentized residuals
        u = u.reshape(residuals.shape)
    else:
        MAD = np.median(AD)  # median absolute deviation
        u = f(residuals /
              (1.4826 * MAD)) / np.sqrt(1 - h)  # studentized residuals

    if method not in _methods:
        raise ValueError(f"{method} is not a supported method: {_methods}")

    if method.lower() == 'cauchy':
        c = 2.385
        weights = 1. / (1 + (u / c) ** 2)
    else:
        raise NotImplementedError("Only 'cauchy' weights are implemented at "
                                  "at the moment")

    weights[np.isnan(weights)] = 0

    return weights


def smoothn(
        data: np.ndarray,
        smooth: Optional[float] = None,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        data_weights: Optional[np.ndarray] = None,
        smoothOrder: Optional[int] = 2,
        init_guess: Optional[np.ndarray] = None,
        isrobust: Optional[bool] = True,
        MaxIter: Optional[int] = 100,
        TolZ: Annotated[float, ValueRange(0.0, 1.0)] = 0.001,
        gap_value: Optional[Union[float, int]] = np.nan,
        exclusion_mask: Optional[np.ndarray] = None,
        data_sampling: Optional[Union[float, Tuple[float, ...]]] = 1.,
        debug_mode: Optional[bool] = False,
        return_stats: Optional[Tuple[str, ...]] = None,
) -> (np.ndarray, float, int, dict):
    """
    Robust spline smoothing for 1-D to 3-D data.
    A fast, automatized and robust discretised smoothing spline for data
    of any dimension.

    When using this algorithm, refer to:
        Garcia, D. (2010) 'Robust smoothing of gridded data in one and higher
        dimensions with missing values',
        Computational Statistics & Data Analysis, 54(4), pp. 1167–1178.
        Available at: https://doi.org/10.1016/j.csda.2009.09.020


    Parameters
    ---------
    data: np.ndarray
        1,2 or 3-dimensional array of values to smoothen.
        data gaps to fill should have the value given as `gap_value`.
    smooth: float, optional (default: None)
        Smoothing parameter.
        If given, smooth must be a real positive scalar.
        The larger smooth is, the smoother the output will be.
        If None is passed smooth is automatically determined from data
        by minimising the generalized cross-validation (GCV) score.
    axis: int or tuple[int,...], optional (default: None)
        Axes along which the smoothing is applied. If not given, smoothing is
        applied along all axes of dat.
    data_weights: np.ndarray, optional (default: None)
        Weight (normally between 0 and 1) for each value in dat.
        Weights must be given as array of same shape as dat.
        Weight 0 would mean that a value is ignored when fitting the model.
        NaNs in data will be assigned the weight 0.
    smoothOrder: int, optional (default: 2)
        Exponential of lambda and gamma used for smoothing.
        1 would be a linear interpolation
    init_guess: np.ndarray, optional (default: None)
        First guess for the interpolated data, resp. initial value for the
        iterative process. Must have the same shape as data if passed.
        If z0 / initial guess is not passed, it will be generated from the
        passed data.
    isrobust: bool, optional (default: False)
        Apply robust weights (not affected by outliers)
    MaxIter:
        Maximum number of iterations allowed (default = 100)
    TolZ:
        Termination tolerance on Z (default = 1e-3)
        TolZ must be in ]0,1[
    exclusion_mask: np.ndarray, optional (default: None)
        A boolean mask of elements in data that are excluded from calculating
        the smoothing function. Must have the same shape as data. Elements
        in data where the corresponding exclusion mask is True will still
        be considered in the interpolation (so that the original distances and
        neighbourhoods are retained), but in the final output they will be
        removed again and replaced with NaNs.
    data_sampling: int or tuple, optional (default: 1)
        Sampling unit for each dimension of ax. If a tuple is given, it must
        be the same length as the number of dimensions of dat. If a single
        number is given, it is applied to all dimensions.
    debug_mode: bool, optional (default: False)
        All debug messages are logged by the pytesmo logger
        (logging.getLogger('pytesmo')). If this setting is activated, we add
        a handler to log to stdout for the DCTPLS function to print the debug
        messages.
    return_stats: tuple, optional (default: None)
        Select which side products should be kept and returned. Removing
        some from the list will reduce memory usage.
            - 'initial_guess':
                    Return the initial guess for smoothing
            - 'euclidean_distance':
                    Return measure for the gap size (only when exclusion
                    mask is set).
            - 'final_weights':
                    How much weight was assigned to each data element in
                    the end.
            - gcv_score:
                    Score of the GCV

    Returns
    -------
    z: np.ndarray
        Smoothed and interpolated version of input data
    smooth: float
        Final parameter that was used
    exit_flag: int
        1: ok
        2: less than 2 data samples
        3: Inner loop did not converge
    stats: dict
        Side products that were selected in `return_stats`
    """

    def _enter_debug_mode():
        logger.setLevel('DEBUG')
        logger.addHandler(streamHandler)

    def _exit_debug_mode():
        logger.setLevel(loglevel)
        logger.handlers.remove(streamHandler)

    if debug_mode:
        _enter_debug_mode()

    exit_flag = 1  # ok
    stats = OrderedDict([
        ('initial_guess', None),
        ('euclidean_distance', None),
        ('final_weights', None),
        ('gcv_score', None),
    ])

    if return_stats is None:
        return_stats = []

    logger.debug(f"Data has shape: {data.shape}")
    # setting up the weights arrays
    if data_weights is not None:
        weights = data_weights.copy()
        if weights.shape != data.shape:
            raise ValueError(f"Data and weights must have the same shape: "
                             f"{data.shape} vs. {weights.shape}")
        weights = (weights / np.max(weights))  # normalise
        logger.debug(f"Weights are found with shape {weights.shape} and "
                     f"normalised with factor {weights.max()}")
    else:
        weights = np.ones(data.shape)
        logger.debug(f"No weights are found. "
                     f"Using 1s with shape {weights.shape}")

    weights = weights.astype(np.float32)

    n_all = data.size

    # Handle case of insufficient data
    _thres = 2

    if n_all < _thres:
        logger.warning(f"Less than {_thres} values to interpolate passed.")
        z = None
        exit_flag = 2
        if debug_mode:
            _exit_debug_mode()
        return z, smooth, exit_flag, stats

    logger.debug(f"Number of elements: {n_all}, which is larger than "
                 f"threshold {_thres}")

    # set data to 0 where we don't want to use it
    isfinite: np.ndarray = \
        (np.isfinite(data) if not np.isfinite(gap_value)
         else data != gap_value).astype(bool) & \
        (True if exclusion_mask is None else ~exclusion_mask)
    # arbitrary values for missing y-data
    data[~isfinite] = 0

    weights[~isfinite] = 0  # also set the weights to 0
    if np.any(weights < 0):
        raise ValueError('Weights must all be >=0')

    logger.debug(
        f"{len(isfinite)} elements in data are NaN and set arbitrarily. "
        f"Weights for those elements set to 0 accordingly")

    isweighted = np.any(weights != 1)
    logger.debug(f"isweighted status to be applied is: {isweighted}")

    autosmooth = True if smooth is None else False
    logger.debug(f"autosmooth status to be applied is: {autosmooth}")

    # Dimensions that the algorithm is applied over
    axis = np.arange(data.ndim) if axis is None else axis
    logger.debug(f"Dimension of data are: {data.ndim}")

    di = 1  # for equally spaced data
    Lambda = np.full(data.shape, 0.).astype(data.dtype)
    for i, ax in enumerate(axis):
        logger.debug(
            f"Lambda (shape {Lambda.shape}) MEAN after {i} "
            f"iterations: {Lambda.mean()}"
        )
        n = int(data.shape[ax])
        _lambda = 2 - 2 * np.cos(np.pi * (np.arange(1, n + 1) - 1.) / n)
        _lambda = _lambda / (di ** 2)
        _shp = [n if a == ax else 1 for a in axis]
        Lambda += _lambda.astype(data.dtype).reshape(_shp)
        del _lambda

    logger.debug(f"Lambda MEAN after scaling: {Lambda.mean()}")
    hMin, hMax = 1e-6, 0.99
    ndim = data.ndim  # np.linalg.matrix_rank(data)

    if smoothOrder == 1:
        sMinBnd = (1 / hMax ** (2 / ndim) - 1) / 4
        sMaxBnd = (1 / hMin ** (2 / ndim) - 1) / 4
    else:
        sMinBnd = (((1 + np.sqrt(1 + 8 * hMax ** (2 / ndim))) / 4 / hMax **
                    (2 / ndim)) ** 2 - 1) / 16
        sMaxBnd = (((1 + np.sqrt(1 + 8 * hMin ** (2 / ndim))) / 4 / hMin **
                    (2 / ndim)) ** 2 - 1) / 16

    logger.debug(
        f"sBands start with min={sMinBnd}, max={sMaxBnd} "
        f"for data with ndim: {ndim}"
    )

    if init_guess is not None:  # initial guess
        z0 = init_guess.copy()
        logger.debug(
            f"z from custom passed initial guess init_guess={z0.shape}")
    else:
        if isweighted:
            s = data_sampling if isinstance(
                data_sampling, Iterable) else tuple(
                np.full(data.ndim, data_sampling))
            z0, dist = calc_init_guess(data, mask=~isfinite, sampling=s)
            logger.debug(f"z from from initial guess init_guess={z0.shape}")
        else:
            z0 = np.full(data.shape, 0).astype(data.dtype)
            logger.debug(f"z from from zeros init_guess={z0.shape}")

    if 'initial_guess' in return_stats:
        stats['initial_guess'] = z0.copy()

    z = z0.copy()

    tol = 1.
    errp = 0.1  # Error on p. Smoothness parameter smooth = 10^p
    # Relaxation factor RF: to speedup convergence
    RF = 1 + 0.75 * bool(isweighted)

    # Main iterative process
    p_best = None
    Wtot = weights.copy()
    score = None

    for i in range(3):  # RobustIterativeProcess, 3 times
        logger.debug(
            f"RobustIterativeProcess {i} -> tol={tol}, errp={errp}, RF={RF}")

        j = 0
        while (tol > TolZ) and j < MaxIter:

            logger.debug(
                f"RobustIterativeProcess {i} -> Inner Loop {j}: tol={tol}")
            assert data.shape == z.shape, "data and Z have differnt shapes"
            logger.debug(
                f"RobustIterativeProcess {i} -> "
                f"data - z={np.sum(np.abs(data - z))}"
            )
            DCTdat = dctNd(Wtot * (data - z) + z, inverse=False)
            logger.debug(
                f"RobustIterativeProcess {i} -> "
                f"Inner Loop {j}: DCTdat mean ={np.nanmean(DCTdat)}"
            )

            # Wtot = None if not weighted else Wtot
            # find the optimal smoothing order for the gcv score

            w = None if (Wtot.sum() / n_all) >= 1. else Wtot
            args = (Lambda, DCTdat, data, smoothOrder, w)

            if autosmooth:
                # to reduce the number of iterations
                if (j == 0) or np.log2(j).is_integer():
                    p_best, fval, ierr, numfunc = \
                        opt.fminbound(
                            gcv, full_output=True, xtol=errp,
                            x1=np.log10(sMinBnd), x2=np.log10(sMaxBnd),
                            args=(*args, True))

                    p_best = p_best.astype(data.dtype)

                    logger.debug(
                        f"RobustIterativeProcess {i} -> "
                        f"Inner Loop {j}: FminBound found best p={p_best}"
                    )
                    logger.debug(
                        f"RobustIterativeProcess {i} -> "
                        f"Inner Loop {j}: Optimise function mid value: {fval}"
                    )
                    logger.debug(
                        f"RobustIterativeProcess {i} -> "
                        f"Inner Loop {j}: Optimise function error flag: {ierr}"
                    )
                    logger.debug(
                        f"RobustIterativeProcess {i} -> "
                        f"Inner Loop {j}: Optimise function calls: {numfunc}"
                    )
                    logger.debug(
                        f"RobustIterativeProcess {i} -> "
                        f"Inner Loop {j}: Autosmooth found best p={p_best}"
                    )
            else:
                p_best = np.log(smooth) / np.log(10)
                logger.debug(
                    f"autosmooth OFF with p_best={p_best} as initial guess")

            # calculate smooth, Gamma and TrH for p_best
            score, smooth, Gamma, TrH = gcv(p_best, *args, score_only=False)

            logger.debug(
                f"RobustIterativeProcess {i} -> Current smooth={smooth}")

            z = RF * dctNd(Gamma * DCTdat, inverse=True) + (1 - RF) * z

            logger.debug(
                f"RobustIterativeProcess {i} -> "
                f"Gamma shape/mean: {Gamma.shape}/{Gamma.mean()}"
            )

            logger.debug(
                f"RobustIterativeProcess {i} -> "
                f"z (shape/mean)={z.shape}/{z.mean()}"
            )

            # if no weighted/missing data => tol=0 (no iteration)
            dz = z[isfinite] - z0[isfinite]
            tol = isweighted * np.linalg.norm(dz) / np.linalg.norm(z[isfinite])
            del dz

            logger.debug(f"RobustIterativeProcess {i} -> tol={tol}")

            z0 = z.copy()  # re-initialization
            logger.debug(f"RobustIterativeProcess {i} -> Reinitialise with z")
            j += 1

        logger.debug(f"--------------------------------------------")
        logger.debug(f"RobustIterativeProcess finished after i={i} "
                     f"tol={tol}")

        if i >= MaxIter and (tol > TolZ):
            exit_flag = 3
            warnings.warn(f"Inner loop DID NOT CONVERGE after {i} iterations")

        if isrobust:  # -- Robust Smoothing: iteratively re-weighted process
            if smoothOrder == 1:
                h = 1 / np.sqrt(1 + 4 * smooth / di ** (2 ** smoothOrder))
            else:
                h = np.sqrt(1 + 16 * smooth / di ** (2 ** smoothOrder))
                h = np.sqrt(1 + h) / np.sqrt(2) / h

            logger.debug(f"Robust smoothing with h={h} (rank={ndim})")

            # --- take robust weights into account
            Wtot = weights * RobustWeights(
                residuals=data - z,
                h=h,
                mask=~isfinite,
                method="cauchy",
                simple=True)

            logger.debug(f"Wtot={Wtot.shape}")

            # --- re-initialize for another iterative weighted process
            isweighted = True
            tol = 1
            # ---
        else:
            logger.debug(f"Not Robust process, escape loop.")
            break

    if 'final_weights' in return_stats:
        stats['final_weights'] = Wtot

    if 'gcv_score' in return_stats:
        stats['gcv_score'] = score

    # Warning messages
    # ---
    if autosmooth:
        if abs(np.log10(smooth) - np.log10(sMinBnd)) < errp:
            warnings.warn(
                f'{smooth}: the LOWER bound for smooth has been reached. '
                'Put smooth as an input variable if required.')
        elif abs(np.log10(smooth) - np.log10(sMaxBnd)) < errp:
            warnings.warn(
                f'{smooth}: the UPPER bound for smooth has been reached. '
                'Put smooth as an input variable if required.')

    # apply the fill values back onto the interpolated data for excluded points
    if exclusion_mask is not None:
        z[exclusion_mask] = gap_value

    if debug_mode:
        _exit_debug_mode()

    return z, smooth, exit_flag, stats
