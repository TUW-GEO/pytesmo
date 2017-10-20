"""
Created on Tue Apr 02 16:50:34 2013

@author: tm

computes julian date, given month (1..12). day(1..31) and year,
its inverse (calendar date from julian), and the day
of the year (doy), assuming it is a leap year.

julday and caldat are adapted from "Numerical Recipes in C', 2nd edition,
pp. 11

restrictions
- no error handling implemented
- works only for years past 1582
- time not yet supported

"""

import numpy as np
import pandas as pd
import datetime as dt
import pytz


def julday(month, day, year, hour=0, minute=0, second=0):
    """
    Julian date from month, day, and year (can be scalars or arrays)

    Parameters
    ----------
    month : numpy.ndarray or int32
        Month.
    day : numpy.ndarray or int32
        Day.
    year : numpy.ndarray or int32
        Year.
    hour : numpy.ndarray or int32, optional
        Hour.
    minute : numpy.ndarray or int32, optional
        Minute.
    second : numpy.ndarray or int32, optional
        Second.



    Returns
    -------
    jul : numpy.ndarray or double
        Julian day.
    """
    month = np.array(month)
    day = np.array(day)
    inJanFeb = month <= 2
    jy = year - inJanFeb
    jm = month + 1 + inJanFeb * 12

    jul = np.int32(np.floor(365.25 * jy) +
                   np.floor(30.6001 * jm) + (day + 1720995.0))
    ja = np.int32(0.01 * jy)
    jul += 2 - ja + np.int32(0.25 * ja)

    jul = jul + hour / 24.0 - 0.5 + minute / 1440.0 + second / 86400.0

    return jul


def caldat(julian):
    """
    Calendar date (month, day, year) from julian date, inverse of 'julday()'
    Return value:  month, day, and year in the Gregorian
    Works only for years past 1582!

    Parameters
    ----------
    julian : numpy.ndarray or double
        Julian day.

    Returns
    -------
    month : numpy.ndarray or int32
        Month.
    day : numpy.ndarray or int32
        Day.
    year : numpy.ndarray or int32
        Year.
    """
    jn = np.int32(((np.array(julian) + 0.000001).round()))

    jalpha = np.int32(((jn - 1867216) - 0.25) / 36524.25)
    ja = jn + 1 + jalpha - (np.int32(0.25 * jalpha))
    jb = ja + 1524
    jc = np.int32(6680.0 + ((jb - 2439870.0) - 122.1) / 365.25)
    jd = np.int32(365.0 * jc + (0.25 * jc))
    je = np.int32((jb - jd) / 30.6001)

    day = jb - jd - np.int32(30.6001 * je)
    month = je - 1
    month = (month - 1) % 12 + 1
    year = jc - 4715
    year = year - (month > 2)

    return month, day, year


def julian2date(julian):
    """
    Calendar date from julian date.
    Works only for years past 1582!

    Parameters
    ----------
    julian : numpy.ndarray or double
        Julian day.

    Returns
    -------
    year : numpy.ndarray or int32
        Year.
    month : numpy.ndarray or int32
        Month.
    day : numpy.ndarray or int32
        Day.
    hour : numpy.ndarray or int32
        Hour.
    minute : numpy.ndarray or int32
        Minute.
    second : numpy.ndarray or int32
        Second.
    """
    min_julian = 2299160
    max_julian = 1827933925

    julian = np.atleast_1d(np.array(julian, dtype=float))

    if np.min(julian) < min_julian or np.max(julian) > max_julian:
        raise ValueError("Value of Julian date is out of allowed range.")

    jn = (np.round(julian + 0.0000001)).astype(np.int32)

    jalpha = (((jn - 1867216) - 0.25) / 36524.25).astype(np.int32)
    ja = jn + 1 + jalpha - (np.int32(0.25 * jalpha))
    jb = ja + 1524
    jc = (6680.0 + ((jb - 2439870.0) - 122.1) / 365.25).astype(np.int32)
    jd = (365.0 * jc + (0.25 * jc)).astype(np.int32)
    je = ((jb - jd) / 30.6001).astype(np.int32)

    day = jb - jd - np.int64(30.6001 * je)
    month = je - 1
    month = (month - 1) % 12 + 1
    year = jc - 4715
    year = year - (month > 2)

    fraction = (julian + 0.5 - jn).astype(np.float64)
    eps = (np.float64(1e-12) * np.abs(jn)).astype(np.float64)
    eps.clip(min=np.float64(1e-12), max=None)
    hour = (fraction * 24. + eps).astype(np.int64)
    hour.clip(min=0, max=23)
    fraction -= hour / 24.
    minute = (fraction * 1440. + eps).astype(np.int64)
    minute = minute.clip(min=0, max=59)
    second = (fraction - minute / 1440.) * 86400.
    second = second.clip(min=0, max=None)
    microsecond = ((second - np.int32(second)) * 1e6).astype(np.int32)
    microsecond = microsecond.clip(min=0, max=999999)
    second = second.astype(np.int32)

    return year, month, day, hour, minute, second, microsecond


def julian2datetime(julian, tz=None):
    """
    converts julian date to python datetime
    default is not time zone aware

    Parameters
    ----------
    julian : float
        julian date
    """
    year, month, day, hour, minute, second, microsecond = julian2date(julian)
    if type(julian) == np.array or type(julian) == np.memmap or \
            type(julian) == np.ndarray or type(julian) == np.flatiter:
        return np.array([dt.datetime(y, m, d, h, mi, s, ms, tz)
                         for y, m, d, h, mi, s, ms in
                         zip(np.atleast_1d(year),
                             np.atleast_1d(month),
                             np.atleast_1d(day),
                             np.atleast_1d(hour),
                             np.atleast_1d(minute),
                             np.atleast_1d(second),
                             np.atleast_1d(microsecond))])

    return dt.datetime(year, month, day, hour, minute, second, microsecond, tz)


def julian2doy(j, consider_nonleap_years=True):
    """
    Calendar date from julian date.
    Works only for years past 1582!

    Parameters
    ----------
    j : numpy.ndarray or double
        Julian days.
    consider_nonleap_years : boolean, optional
        Flag if all dates are interpreted as leap years (False) or not (True).

    Returns
    -------
    doy : numpy.ndarray or int32
        Day of year.
    """
    year, month, day = julian2date(j)[0:3]

    if consider_nonleap_years:
        return doy(month, day, year)
    else:
        return doy(month, day)


def julian2datetimeindex(j, tz=pytz.UTC):
    """
    Converting Julian days to datetimeindex.

    Parameters
    ----------
    j : numpy.ndarray or int32
        Julian days.
    tz : instance of pytz, optional
        Time zone. Default: UTC

    Returns
    -------
    datetime : pandas.DatetimeIndex
        Datetime index.
    """
    year, month, day, hour, minute, second, microsecond = julian2date(j)

    return pd.DatetimeIndex([dt.datetime(y, m, d, h, mi, s, ms, tz)
                             for y, m, d, h, mi, s, ms in
                             zip(year, month, day, hour, minute,
                                 second, microsecond)])


def julian2num(j):
    """
    Convert a matplotlib date to a Julian days.

    Parameters
    ----------
    j : numpy.ndarray : int32
        Julian days.

    Returns
    -------
    num : numpy.ndarray : int32
        Number of days since 0001-01-01 00:00:00 UTC *plus* *one*.
    """
    return j - 1721424.5


def num2julian(n):
    """
    Convert a Julian days to a matplotlib date.

    Parameters
    ----------
    n : numpy.ndarray : int32
        Number of days since 0001-01-01 00:00:00 UTC *plus* *one*.

    Returns
    -------
    j : numpy.ndarray : int32
        Julian days.
    """
    return n + 1721424.5


def doy(month, day, year=None):
    """
    Calculation of day of year. If year is provided it will be tested for
    leap years.

    Parameters
    ----------
    month : numpy.ndarray or int32
        Month.
    day : numpy.ndarray or int32
        Day.
    year : numpy.ndarray or int32, optional
        Year.

    Retruns
    -------
    doy : numpy.ndarray or int32
        Day of year.
    """
    daysPast = np.array([0, 31, 60, 91, 121, 152, 182, 213,
                         244, 274, 305, 335, 366])

    day_of_year = daysPast[month - 1] + day

    if year is not None:
        nonleap_years = np.invert(is_leap_year(year))
        day_of_year = (day_of_year -
                       nonleap_years.astype('int') +
                       np.logical_and(day_of_year < 60, nonleap_years).astype('int'))

    return day_of_year


def is_leap_year(year):
    """
    Check if year is a leap year.

    Parameters
    ----------
    year : numpy.ndarray or int32

    Returns
    -------
    leap_year : numpy.ndarray or boolean
        True if year is a leap year.
    """
    return np.logical_or(np.logical_and(year % 4 == 0, year % 100 != 0),
                         year % 400 == 0)
