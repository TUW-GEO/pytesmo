'''
Created on June 20, 2013
'''
import warnings

import pandas as pd
import numpy as np
from cadati.conv_doy import doy
from cadati.jd_date import julian2date
from pytesmo.time_series.filtering import moving_average


def calc_anomaly(Ser,
                 window_size=35,
                 climatology=None,
                 respect_leap_years=True,
                 return_clim=False):
    '''
    Calculates the anomaly of a time series (Pandas series).
    Both, climatology based, or moving-average based anomalies can be
    calculated

    Parameters
    ----------
    Ser : pandas.Series
        Input data (index must be a DateTimeIndex)
    window_size : float, optional (default: 35)
        The window-size [days] of the moving-average window to calculate the
        anomaly reference (only used if climatology is not provided)
    climatology : pandas.Series (index: 1-366), optional (default: None)
        if provided, anomalies will be based on the climatology
    timespan : [timespan_from, timespan_to], datetime.datetime(y,m,d), optional
        If set, only a subset
    respect_leap_years : boolean, optional (default: True)
        If set then leap years will be respected during matching of the
        climatology to the time series
    return_clim : boolean, optional (default: False)
        if set to true the return argument will be a DataFrame which
        also contains the climatology time series.
        Only has an effect if climatology is used.

    Returns
    -------
    anomaly : pandas.Series or pandas.DataFrame
        Series containing the calculated anomalies.
        If `return_clim` is set to true, a DataFrame will be returned, where
        one column contains the anomalies and another the climatology
        broadcasted over the whole index. If a climatology with a 'std' column
        was passed initially, this column will also be returned in the
        DataFrame if `return_clim` is chosen.
    '''

    if climatology is not None:

        if type(Ser.index) == pd.DatetimeIndex:

            year, month, day = (np.asarray(Ser.index.year),
                                np.asarray(Ser.index.month),
                                np.asarray(Ser.index.day))

        else:
            year, month, day = julian2date(Ser.index.values)[0:3]

        if respect_leap_years:
            doys = doy(month, day, year)
        else:
            doys = doy(month, day)

        df = pd.DataFrame()
        df['absolute'] = Ser
        df['doy'] = doys

        if isinstance(climatology, pd.Series):
            clim = pd.DataFrame({'climatology': climatology})
        else:
            clim = climatology

        df = df.join(clim, on='doy', how='left')

        anomaly = df['absolute'] - df['climatology']
        anomaly.index = df.index

        if return_clim:
            anomaly = pd.DataFrame({'anomaly': anomaly})
            anomaly['climatology'] = df['climatology']
            if 'std' in df.columns:
                anomaly['climatology_std'] = df['std']

    else:
        reference = moving_average(Ser, window_size=window_size)
        anomaly = Ser - reference

    return anomaly


def _index_units(year,
                 month,
                 day,
                 unit="day",
                 respect_leap_years=True) -> (np.array, int):
    """Provide the indices for the specified unit type and the index lenth"""
    if respect_leap_years:
        args = month, day, year
    else:
        args = month, day

    if unit == "day":
        return doy(*args), 366
    elif unit == "month":
        return month, 12
    else:
        raise ValueError(f"Invalid unit: {unit}")


def calc_climatology(Ser,
                     moving_avg_orig=5,
                     moving_avg_clim=None,
                     median=False,
                     std=False,
                     timespan=None,
                     fill=np.nan,
                     wraparound=True,
                     respect_leap_years=False,
                     interpolate_leapday=False,
                     fillna=True,
                     min_obs_orig=1,
                     min_obs_clim=1,
                     output_freq="day"):
    """
    Calculates the climatology of a data set.

    Parameters
    ----------
    Ser : pandas.Series
        Time series to compute climatology for (index must be a
        DateTimeIndex or julian date)
    moving_avg_orig : float, optional (default: 5)
        The size of the moving_average window [days] that will be applied on
        the input Series (gap filling, short-term rainfall correction)
    moving_avg_clim : float, optional (default: None)
        The size of the moving_average window in days that will be applied on
        the calculated climatology (long-term event correction).
        If None is passed, it will be calculated from the 'output_freq' value:
            - 'day': 35
            - 'month': 3
    median : boolean, optional (default: False)
        if set to True, the climatology will be based on the median conditions
    std: boolean, optional (default: False)
        if set to True, there will be 2 columns, one for the median or mean
        and one of the standard deviation of the aggregated data points.
    timespan : [timespan_from, timespan_to], datetime.datetime(y,m,d), optional
        Set this to calculate the climatology based on a subset of the input
        Series
    fill : float or int, optional (default: np.nan)
        Fill value to use for days on which no climatology exists
    wraparound : boolean, optional (default: True)
        If set then the climatology is wrapped around at the edges before
        doing the second running average (long-term event correction)
    respect_leap_years : boolean, optional (default: False)
        If set then leap years will be respected during the calculation of
        the climatology. Only valid with 'output_freq' value set to 'day'.
        Default: False
    interpolate_leapday: boolean, optional (default: False)
        <description>. Only valid with 'output_freq' value set to 'day'.
        Default: False
    fillna: boolean, optional (default: True)
        If set, then the moving average used for the calculation of the
        climatology will be filled at the nan-values
    min_obs_orig: int (default: 1)
        Minimum observations required to give a valid output in the first
        moving average applied on the input series
    min_obs_clim: int (default: 1)
        Minimum observations required to give a valid output in the second
        moving average applied on the calculated climatology
    output_freq: str, optional (default: 'day')
        Determines the output frequency (time unit) of the climatology
        calculation (independently of the 'Ser' input frequency).
        Currently, supported options are 'day', 'month'.

    Returns
    -------
    climatology : pandas.Series or pandas.DataFrame
        Containing the calculated climatology. The size of the series depends
        on the type of climatology being calculated, based on the value of
        'output_freq':
            - 366 values for a daily climatology, behaving as a leap year
            - 12 values for a monthly climatology
        If 'std' is set to True, the output will be a DataFrame with 2 columns:
            'climatology' and 'std'.
    """
    # establish the moving window size
    default_moving_avg_clim = {"day": 35, "month": 3}

    if moving_avg_clim is None:
        moving_avg_clim = default_moving_avg_clim[output_freq]

    if output_freq == "month" and moving_avg_clim > 5:
        # in case someone changes unit but not moving_avg_clim a warning
        # might be useful
        warnings.warn(f"Window for moving average of climatology set to "
                      f"{moving_avg_clim} months, is this intentional?")

    if output_freq != "day":
        # irrelevant at lower frequencies than daily
        respect_leap_years, interpolate_leapday = False, False

    if timespan is not None:
        Ser = Ser.truncate(before=timespan[0], after=timespan[1])

    Ser = moving_average(
        Ser, window_size=moving_avg_orig, fillna=fillna, min_obs=min_obs_orig)

    Ser = pd.DataFrame(Ser)

    if type(Ser.index) == pd.DatetimeIndex:
        year, month, day = (np.asarray(Ser.index.year),
                            np.asarray(Ser.index.month),
                            np.asarray(Ser.index.day))
    else:
        year, month, day = julian2date(Ser.index.values)[0:3]

    # provide indices for the selected unit
    indices, n_idx = _index_units(
        year,
        month,
        day,
        unit=output_freq,
        respect_leap_years=respect_leap_years)
    Ser['unit'] = indices

    if median:
        clim = Ser.groupby('unit').median()
        clim.name = 'climatology'
    else:
        clim = Ser.groupby('unit').mean()
        clim.name = 'climatology'

    if std:
        std_ser = Ser.groupby('unit').std()

        clim_ser = pd.concat([
            clim.loc[:, 0].rename(clim.name),
            std_ser.loc[:, 0].rename('std')], axis=1)  # yapf: disable
    else:
        clim_ser = pd.DataFrame(
            data={'climatology': clim.values.flatten()},
            index=clim.index.values)

    clim_ser = clim_ser.reindex(np.arange(n_idx) + 1)

    if wraparound:
        index_old = clim_ser.index.copy()
        left_mirror = clim_ser.iloc[-moving_avg_clim:]
        right_mirror = clim_ser.iloc[:moving_avg_clim]
        # Shift index to start at n_idx - index at -moving_avg_clim
        # to run over a whole year while keeping gaps the same size
        right_mirror.index = right_mirror.index + n_idx * 2
        clim_ser.index = clim_ser.index + n_idx
        clim_ser = pd.concat([left_mirror, clim_ser, right_mirror])

    clim_ser['climatology'] = moving_average(
        clim_ser['climatology'],
        window_size=moving_avg_clim,
        fillna=fillna,
        min_obs=min_obs_clim)

    if wraparound:
        clim_ser = clim_ser.iloc[moving_avg_clim:-moving_avg_clim]
        clim_ser.index = index_old

    # keep hardcoding as it's only for doys
    if interpolate_leapday and not respect_leap_years:
        clim_ser.loc[60, :] = np.mean(
            (clim_ser.loc[59, :], clim_ser.loc[61, :]))
    elif interpolate_leapday and respect_leap_years:
        clim_ser.loc[366, :] = np.mean(
            (clim_ser.loc[365, :], clim_ser.loc[1, :]))

    clim_ser = clim_ser.fillna(fill)

    if len(clim_ser.columns) == 1:
        return clim_ser.iloc[:, 0]
    else:
        return clim_ser
