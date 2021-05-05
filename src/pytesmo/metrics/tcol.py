"""
Triple collocation metrics.

To avoid recalculation of the covariance matrix, all metrics are calculated
inside :py:func:`tcol_metrics`. Bootstrapped CIs can be obtained with
:py:func:`tcol_metrics_with_bootstrapped_ci`.

Bootstrapping is currently not available for extended TCA.
"""


from itertools import permutations, combinations
import numpy as np


@np.errstate(invalid="ignore")
def tcol_metrics(x, y, z, ref_ind=0):
    """
    Triple collocation based estimation of signal-to-noise ratio, absolute
    errors, and rescaling coefficients

    Parameters
    ----------
    x: 1D numpy.ndarray
        first input dataset
    y: 1D numpy.ndarray
        second input dataset
    z: 1D numpy.ndarray
        third input dataset
    ref_ind: int
        Index of reference data set for estimating scaling
        coefficients. Default: 0 (x)

    Returns
    -------
    snr: numpy.ndarray
        signal-to-noise (variance) ratio [dB]
    err_std: numpy.ndarray
        **SCALED** error standard deviation
    beta: numpy.ndarray
         scaling coefficients (i_scaled = i * beta_i)

    Notes
    -----

    This function estimates the triple collocation errors, the scaling
    parameter :math:`\\beta` and the signal to noise ratio directly from the
    covariances of the dataset. For a general overview and how this function
    and :py:func:`pytesmo.metrics.tcol_error` are related please see
    [Gruber2015]_.

    Estimation of the error variances from the covariances of the datasets
    (e.g. :math:`\\sigma_{XY}` for the covariance between :math:`x` and
    :math:`y`) is done using the following formula:

    .. math::

       \\sigma_{\\varepsilon_x}^2 =
           \\sigma_{X}^2 - \\frac{\\sigma_{XY}\\sigma_{XZ}}{\\sigma_{YZ}}
    .. math::

       \\sigma_{\\varepsilon_y}^2 =
           \\sigma_{Y}^2 - \\frac{\\sigma_{YX}\\sigma_{YZ}}{\\sigma_{XZ}}
    .. math::

       \\sigma_{\\varepsilon_z}^2 =
           \\sigma_{Z}^2 - \\frac{\\sigma_{ZY}\\sigma_{ZX}}{\\sigma_{YX}}

    :math:`\\beta` can also be estimated from the covariances:

    .. math:: \\beta_x = 1
    .. math:: \\beta_y = \\frac{\\sigma_{XZ}}{\\sigma_{YZ}}
    .. math:: \\beta_z=\\frac{\\sigma_{XY}}{\\sigma_{ZY}}

    The signal to noise ratio (SNR) is also calculated from the variances
    and covariances:

    .. math::

       \\text{SNR}_X[dB] = -10\\log\\left(\\frac{\\sigma_{X}^2\\sigma_{YZ}}
                                         {\\sigma_{XY}\\sigma_{XZ}}-1\\right)
    .. math::

       \\text{SNR}_Y[dB] = -10\\log\\left(\\frac{\\sigma_{Y}^2\\sigma_{XZ}}
                                         {\\sigma_{YX}\\sigma_{YZ}}-1\\right)
    .. math::

       \\text{SNR}_Z[dB] = -10\\log\\left(\\frac{\\sigma_{Z}^2\\sigma_{XY}}
                                         {\\sigma_{ZX}\\sigma_{ZY}}-1\\right)

    It is given in dB to make it symmetric around zero. If the value is zero
    it means that the signal variance and the noise variance are equal. +3dB
    means that the signal variance is twice as high as the noise variance.

    References
    ----------
    .. [Gruber2015] Gruber, A., Su, C., Zwieback, S., Crow, W., Dorigo, W.,\
    Wagner, W.  (2015). Recent advances in (soil moisture) triple\
    collocation analysis.  International Journal of Applied Earth\
    Observation and Geoinformation, in review
    """

    cov = np.cov(np.vstack((x, y, z)))
    return _tcol_metrics_from_cov(cov, ref_ind)


def _tcol_metrics_from_cov(cov, ref_ind=0):
    """
    Calculate TCA metrics from pre-computed covariance matrix
    """
    ind = (0, 1, 2, 0, 1, 2)
    no_ref_ind = np.where(np.arange(3) != ref_ind)[0]

    # we have to take the absolute value of the ratio of covariances and of the
    # full difference according to Gruber et al. 2020, Eq. 11
    snr = 10 * np.log10(
        [
            np.abs(
                np.abs((cov[i, i] * cov[ind[i + 1], ind[i + 2]])
                       / (cov[i, ind[i + 1]] * cov[i, ind[i + 2]]))
                - 1
            )
            ** (-1)
            for i in np.arange(3)
        ]
    )
    err_var = np.array(
        [
            cov[i, i]
            - (cov[i, ind[i + 1]] * cov[i, ind[i + 2]])
            / cov[ind[i + 1], ind[i + 2]]
            for i in np.arange(3)
        ]
    )

    beta = np.array(
        [
            cov[ref_ind, no_ref_ind[no_ref_ind != i][0]]
            / cov[i, no_ref_ind[no_ref_ind != i][0]]
            if i != ref_ind
            else 1
            for i in np.arange(3)
        ]
    )

    return snr, np.sqrt(err_var) * beta, beta


def check_if_biased(combs, correlated):
    """
    Supporting function for extended collocation
    Checks whether the estimators are biased by checking of not
    too manny data sets (are assumed to have cross-correlated errors)
    """
    for corr in correlated:
        for comb in combs:
            if (np.array_equal(comb, corr)) | (
                np.array_equal(comb, corr[::-1])
            ):
                return True
    return False


def ecol(data, correlated=None, err_cov=None, abs_est=True):
    """
    Extended collocation analysis to obtain estimates of:

        - signal variances
        - error variances
        - signal-to-noise ratios [dB]
        - error cross-covariances (and -correlations)

    based on an arbitrary number of N>3 data sets.

    **EACH DATA SET MUST BE MEMBER OF >= 1 TRIPLET THAT FULFILLS THE CLASSICAL
    TRIPLE COLLOCATION ASSUMPTIONS**

    Parameters
    ----------
    data : pd.DataFrame
        Temporally matched input data sets in each column
    correlated : tuple of tuples (string)
        A tuple containing tuples of data set names (column names), between
        which the error cross-correlation shall be estimated.
        e.g. [['AMSR-E','SMOS'],['GLDAS','ERA']] estimates error
        cross-correlations between (AMSR-E and SMOS), and (GLDAS and ERA),
        respectively.
    err_cov :
        A priori known error cross-covariances that shall be included
        in the estimation (to obtain unbiased estimates)
    abs_est :
        Force absolute values for signal and error variance estimates
        (to mitiate the issue of estimation uncertainties)

    Returns
    -------
    A dictionary with the following entries (<name> correspond to data set (df
    column's) names:
    - sig_<name> : signal variance of <name>
    - err_<name> : error variance of <name>
    - snr_<name> : SNR (in dB) of <name>
    - err_cov_<name1>_<name2> : error covariance between <name1> and <name2>
    - err_corr_<name1>_<name2> : error correlation between <name1> and <name2>

    Notes
    -----
    Rescaling parameters can be derived from the signal variances
    e.g., scaling <src> against <ref>:
    beta =  np.sqrt(sig_<ref> / sig_<src>)
    rescaled = (data[<src>] - data[<src>].mean()) * beta + data[<ref>].mean()

    Examples
    --------
    .. code-block::

        # Just random numbers for demonstrations
        ds1 = np.random.normal(0,1,500)
        ds2 = np.random.normal(0,1,500)
        ds3 = np.random.normal(0,1,500)
        ds4 = np.random.normal(0,1,500)
        ds5 = np.random.normal(0,1,500)

        # Three data sets without cross-correlated errors: This is equivalent
        # to standard triple collocation.
        df = pd.DataFrame({'ds1':ds1,'ds2':ds2,'ds3':ds3},
                          index=np.arange(500))
        res = ecol(df)

        # Five data sets, where data sets (1 and 2), and (3 and 4), are assumed
        # to have cross-correlated errors.
        df = pd.DataFrame({'ds1':ds1,'ds2':ds2,'ds3':ds3,'ds4':ds4,'ds5':ds5},
                          index=np.arange(500),)
        correlated = [['ds1','ds2'],['ds3','ds4']]
        res = ecol(df,correlated=correlated)

    References
    ----------
    .. [Gruber2016] Gruber, A., Su, C. H., Crow, W. T., Zwieback, S., Dorigo,\
    W. A., & Wagner, W. (2016). Estimating error cross-correlations in soil\
    moisture data sets using extended collocation analysis. Journal of\
    Geophysical Research: Atmospheres, 121(3), 1208-1219.
    """
    data.dropna(inplace=True)

    cols = data.columns.values
    cov = data.cov()

    # subtract a-priori known error covariances to obtained unbiased estimates
    if err_cov is not None:
        cov[err_cov[0]][err_cov[1]] -= err_cov[2]
        cov[err_cov[1]][err_cov[0]] -= err_cov[2]

    # ----- Building up the observation vector y and the design matrix A -----

    # Initial lenght of the parameter vector x:
    # n error variances + n signal variances
    n_x = 2 * len(cols)

    # First n elements in y: variances of all data sets
    y = cov.values.diagonal()

    # Extend y if data sets with correlated errors exist
    if correlated is not None:
        # additionally estimated in x:
        # k error covariances, and k cross-biased signal variances
        # (biased with the respective beta_i*beta_j)
        n_x += 2 * len(correlated)

        # add covariances between the correlated data sets to the y vector
        y = np.hstack((y, [cov[ds[0]][ds[1]] for ds in correlated]))

    # Generate the first part of the design matrix A (total variance = signal
    # variance + error variance)
    A = np.hstack(
        (
            np.matrix(np.identity(int(n_x / 2))),
            np.matrix(np.identity(int(n_x / 2))),
        )
    ).astype("int")

    # build up A and y components for estimating signal variances (biased with
    # beta_i**2 only)
    # i.e., the classical TC based signal variance estimators
    # cov(a,c)*cov(a,d)/cov(c,d)
    for col in cols:

        others = cols[cols != col]
        combs = list(combinations(others, 2))

        for comb in combs:
            if correlated is not None:
                if check_if_biased(
                    [[col, comb[0]], [col, comb[1]], [comb[0], comb[1]]],
                    correlated,
                ):
                    continue

            A_line = np.zeros(n_x).astype("int")
            A_line[np.where(cols == col)[0][0]] = 1
            A = np.vstack((A, A_line))

            y = np.append(
                y,
                cov[col][comb[0]] * cov[col][comb[1]] / cov[comb[0]][comb[1]],
            )

    # build up A and y components for the cross-biased signal variabilities
    # (with beta_i*beta_j)
    # i.e., the cross-biased signal variance estimators
    # (cov(a,c)*cov(b,d)/cov(c,d))
    if correlated is not None:
        for i in np.arange(len(correlated)):
            others = cols[
                (cols != correlated[i][0]) & (cols != correlated[i][1])
            ]
            combs = list(permutations(others, 2))

            for comb in combs:
                if check_if_biased(
                    [
                        [correlated[i][0], comb[0]],
                        [correlated[i][1], comb[1]],
                        comb,
                    ],
                    correlated,
                ):
                    continue

                A_line = np.zeros(n_x).astype("int")
                A_line[len(cols) + i] = 1
                A = np.vstack((A, A_line))

                y = np.append(
                    y,
                    cov[correlated[i][0]][comb[0]]
                    * cov[correlated[i][1]][comb[1]]
                    / cov[comb[0]][comb[1]],
                )

    y = np.matrix(y).T

    # ----- Solving for the parameter vector x -----

    x = (A.T * A).I * A.T * y
    x = np.squeeze(np.array(x))

    # ----- Building up the result dictionary -----

    tags = np.hstack(("sig_" + cols, "err_" + cols, "snr_" + cols))

    if correlated is not None:

        # remove the cross-biased signal variabilities (with beta_i*beta_j) as
        # they are not useful
        x = np.delete(x, np.arange(len(correlated)) + len(cols))

        # Derive error cross-correlations from error covariances and error
        # variances
        for i in np.arange(len(correlated)):
            x = np.append(
                x,
                x[2 * len(cols) + i]
                / np.sqrt(
                    x[len(cols) + np.where(cols == correlated[i][0])[0][0]]
                    * x[len(cols) + np.where(cols == correlated[i][1])[0][0]]
                ),
            )

        # add error covariances and correlations to the result dictionary
        tags = np.append(
            tags,
            np.array(["err_cov_" + ds[0] + "_" + ds[1] for ds in correlated]),
        )
        tags = np.append(
            tags,
            np.array(["err_corr_" + ds[0] + "_" + ds[1] for ds in correlated]),
        )

    # force absolute signal and error variance estimates to compensate for
    # estimation uncertainties
    if abs_est is True:
        x[0:(2 * len(cols))] = np.abs(x[0:(2 * len(cols))])

    # calculate and add SNRs (in decibel units)
    x = np.insert(
        x,
        2 * len(cols),
        10 * np.log10(x[0:len(cols)] / x[len(cols):(2 * len(cols))]),
    )

    return dict(zip(tags, x))
