'''
Created on June 20, 2013

@author: Alexander Gruber Alexander.Gruber@geo.tuwien.ac.at
'''

import numpy as np
import pandas as pd
import datetime
from pytesmo.timedate.julian import doy, julian2date
from pytesmo.time_series.filtering import moving_average

def calc_anomaly(Ser,
                 window_size=35,
                 climatology=None):
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

    timespann : [timespan_from, timespan_to], datetime.datetime(y,m,d), optional
        If set, only a subset

    Returns
    -------
    anomaly : pandas.Series
        Series containing the calculated anomalies
    '''

    if climatology is not None:

        if type(Ser.index) == pd.DatetimeIndex:

            doys = doy(Ser.index.month, Ser.index.day)

        else:
            year, month, day = julian2date(Ser.index.values)[0:3]
            doys = doy(month, day)

        df = pd.DataFrame()
        df['absolute'] = Ser
        df['doy'] = doys

        clim = pd.DataFrame(climatology, columns=['climatology'])

        df = df.join(clim, on='doy', how='left')

        anomaly = df['absolute'] - df['climatology']
        anomaly.index = df.index


    else:
        reference = moving_average(Ser, window_size=window_size)
        anomaly = Ser - reference

    return anomaly


def calc_climatology(Ser,
                     moving_avg_orig=5,
                     moving_avg_clim=30,
                     median=False,
                     timespan=None):
    '''
    Calculates the climatology of a data set

    Parameters
    ----------
    Ser : pandas.Series (index must be a DateTimeIndex or julian date)

    moving_avg_orig : float, optional
        The size of the moving_average window [days] that will be applied on the
        input Series (gap filling, short-term rainfall correction)
        Default: 5

    moving_avg_clim : float, optional
        The size of the moving_average window [days] that will be applied on the
        calculated climatology (long-term event correction)
        Default: 35

    median : boolean, optional
        if set to True, the climatology will be based on the median conditions

    timespan : [timespan_from, timespan_to], datetime.datetime(y,m,d), optional
        Set this to calculate the climatology based on a subset of the input
        Series

    Returns
    -------
    climatology : pandas.Series
        Series containing the calculated climatology
    '''

    if timespan is not None:
        Ser = Ser.truncate(before=timespan[0], after=timespan[1])

    Ser = moving_average(Ser, window_size=moving_avg_orig)

    Ser = pd.DataFrame(Ser)

    if type(Ser.index) == pd.DatetimeIndex:

        doys = doy(Ser.index.month, Ser.index.day)

    else:
        year, month, day = julian2date(Ser.index.values)[0:3]
        doys = doy(month, day)


    Ser['doy'] = doys


    if median:
        clim = Ser.groupby('doy').median()
    else:
        clim = Ser.groupby('doy').mean()

    return moving_average(pd.Series(clim.values.flatten(), index=clim.index.values), window_size=moving_avg_clim)
