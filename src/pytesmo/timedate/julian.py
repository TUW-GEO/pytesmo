"""
Most datetime functions from pytesmo are now in the cadati package
Website: https://github.com/TUW-GEO/cadati
To install cadati, run 'pip install cadati'
This module will be removed in future versions of pytesmo
"""

import pandas as pd
import datetime as dt
import pytz

import warnings

from cadati.check_date import is_leap_year
from cadati.jd_date import (
    julday,
    caldat,
    julian2date,
    julian2datetime,
    julian2num,
    num2julian,
)
from cadati.conv_doy import doy
from pytesmo.utils import deprecated

# deprecated calls:
__all__ = ['is_leap_year', 'julday', 'caldat', 'julian2datetime',
           'julian2num', 'num2julian', 'doy']

warnings.warn(
    "pytesmo.timdate.julian has moved to the cadati pip package "
    "Import from pytesmo will not work in future versions of the package",
    DeprecationWarning,
)


@deprecated()
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
    return julian2date(
        j, return_doy=True, doy_leap_year=not consider_nonleap_years
    )[-1]


@deprecated()
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

    return pd.DatetimeIndex(
        [
            dt.datetime(y, m, d, h, mi, s, ms, tz)
            for y, m, d, h, mi, s, ms in zip(
                year, month, day, hour, minute, second, microsecond
            )
        ]
    )
