'''
Created on June 20, 2013
'''

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
    Ser : pandas.Series (index must be a DateTimeIndex)

    window_size : float, optional
        The window-size [days] of the moving-average window to calculate the
        anomaly reference (only used if climatology is not provided)
        Default: 35 (days)

    climatology : pandas.Series (index: 1-366), optional
        if provided, anomalies will be based on the climatology

    timespan : [timespan_from, timespan_to], datetime.datetime(y,m,d), optional
        If set, only a subset

    respect_leap_years : boolean, optional
        If set then leap years will be respected during matching of the climatology
        to the time series

    return_clim : boolean, optional
        if set to true the return argument will be a DataFrame which
        also contains the climatology time series.
        Only has an effect if climatology is used.

    Returns
    -------
    anomaly : pandas.Series
        Series containing the calculated anomalies
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

        clim = pd.DataFrame({'climatology': climatology})

        df = df.join(clim, on='doy', how='left')

        anomaly = df['absolute'] - df['climatology']
        anomaly.index = df.index

        if return_clim:
            anomaly = pd.DataFrame({'anomaly': anomaly})
            anomaly['climatology'] = df['climatology']

    else:
        reference = moving_average(Ser, window_size=window_size)
        anomaly = Ser - reference

    return anomaly


def _index_units(year, month, day, unit="day", respect_leap_years=True) -> (np.array, int):
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
                     moving_avg_clim=35,
                     moving_avg_month_clim=3,
                     median=False,
                     timespan=None,
                     fill=np.nan,
                     wraparound=True,
                     respect_leap_years=False,
                     interpolate_leapday=False,
                     fillna=True,
                     min_obs_orig=1,
                     min_obs_clim=1,
                     unit="day"):
    """
    Calculates the climatology of a data set.

    Parameters
    ----------
    Ser : pandas.Series (index must be a DateTimeIndex or julian date)

    moving_avg_orig : float, optional
        The size of the moving_average window [days] that will be applied on the
        input Series (gap filling, short-term rainfall correction)
        Default: 5

    moving_avg_clim : float, optional
        The size of the moving_average window in days that will be applied on the
        calculated climatology (long-term event correction)
        Default: 35

    moving_avg_month_clim: : float, optional
        Same as for 'moving_avg_clim', but applied to monthly climatologies. In case
        unit='month', this value overrides 'moving_avg_clim'
        Default: 3

    median : boolean, optional
        if set to True, the climatology will be based on the median conditions

    timespan : [timespan_from, timespan_to], datetime.datetime(y,m,d), optional
        Set this to calculate the climatology based on a subset of the input
        Series

    fill : float or int, optional
        Fill value to use for days on which no climatology exists

    wraparound : boolean, optional
        If set then the climatology is wrapped around at the edges before
        doing the second running average (long-term event correction)

    respect_leap_years : boolean, optional
        If set then leap years will be respected during the calculation of
        the climatology. Only valid with 'unit' value set to 'day'.
        Default: False

    interpolate_leapday: boolean, optional
        <description>. Only valid with 'unit' value set to 'day'.
        Default: False

    fillna: boolean, optional
        If set, then the moving average used for the calculation of the
        climatology will be filled at the nan-values

    min_obs_orig: int
        Minimum observations required to give a valid output in the first
        moving average applied on the input series

    min_obs_clim: int
        Minimum observations required to give a valid output in the second
        moving average applied on the calculated climatology

    unit: str, optional
        Unit of the year to apply the climatology calculation to. Currently,
        supported options are 'day', 'month'.
        Default: 'day'

    Returns
    -------
    climatology : pandas.Series
        Series containing the calculated climatology
        Always has 366 values behaving like a leap year
    """
    if unit != "day":
        respect_leap_years, interpolate_leapday = False, False

    if unit == "month":
        moving_avg_clim = moving_avg_month_clim

    if timespan is not None:
        Ser = Ser.truncate(before=timespan[0], after=timespan[1])

    Ser = moving_average(Ser, window_size=moving_avg_orig, fillna=fillna, min_obs=min_obs_orig)

    Ser = pd.DataFrame(Ser)

    if type(Ser.index) == pd.DatetimeIndex:
        year, month, day = (np.asarray(Ser.index.year),
                            np.asarray(Ser.index.month),
                            np.asarray(Ser.index.day))
    else:
        year, month, day = julian2date(Ser.index.values)[0:3]

    # provide indices for the selected unit
    indices, n_idx = _index_units(
        year, month, day,
        unit=unit,
        respect_leap_years=respect_leap_years
    )
    Ser['unit'] = indices

    if median:
        clim = Ser.groupby('unit').median()
    else:
        clim = Ser.groupby('unit').mean()

    clim_ser = pd.Series(clim.values.flatten(),
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
        clim_ser = pd.concat([left_mirror,
                              clim_ser,
                              right_mirror])

        clim_ser = moving_average(clim_ser, window_size=moving_avg_clim, fillna=fillna, min_obs=min_obs_clim)
        clim_ser = clim_ser.iloc[moving_avg_clim:-moving_avg_clim]
        clim_ser.index = index_old
    else:
        clim_ser = moving_average(clim_ser, window_size=moving_avg_clim, fillna=fillna, min_obs=min_obs_clim)

    # keep hardcoding as it's only for doys
    if interpolate_leapday and not respect_leap_years:
        clim_ser[60] = np.mean((clim_ser[59], clim_ser[61]))
    elif interpolate_leapday and respect_leap_years:
        clim_ser[366] = np.mean((clim_ser[365], clim_ser[1]))

    clim_ser = clim_ser.fillna(fill)

    return clim_ser
