# Copyright (c) 2014, Vienna University of Technology (TU Wien), Department
# of Geodesy and Geoinformation (GEO).
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of the Vienna University of Technology,
#      Department of Geodesy and Geoinformation nor the
#      names of its contributors may be used to endorse or promote products
#      derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL VIENNA UNIVERSITY OF TECHNOLOGY,
# DEPARTMENT OF GEODESY AND GEOINFORMATION BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


"""
Module provides grouping functions that can be used together with pandas
to create a few strange timegroupings like e.g. decadal products were
there are three products per month with timestamps on the 10th 20th and last
of the month
"""
from dataclasses import dataclass
from typing import Optional, Union, Tuple, List

import pandas as pd
import numpy as np
from datetime import date, datetime
import calendar

from cadati.conv_doy import doy


def group_by_day_bin(df, bins=[1, 11, 21, 32], start=False,
                     dtindex=None):
    """
    Calculates timegroups for a given daterange. Groups are from day
    1-10,
    11-20,
    21-last day of each month.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with DateTimeIndex for which the grouping should be
        done
    bins : list, optional
        bins in day of the month, default is for dekadal grouping
    start : boolean, optional
        if set to True the start of the bin will be the timestamp for each
        observations
    dtindex : pandas.DatetimeIndex, optional
        precomputed DatetimeIndex that should be used for resulting
        groups, useful for processing of numerous datasets since it does not
        have to be computed for every call

    Returns
    -------
    grouped : pandas.core.groupby.DataFrameGroupBy
        DataFrame groupby object according the the day bins
        on this object functions like sum() or mean() can be
        called to get the desired aggregation.
    dtindex : pandas.DatetimeIndex
        returned so that it can be reused if possible
    """

    dekads = np.digitize(df.index.day, bins)
    if dtindex is None:
        dtindex = grp_to_datetimeindex(dekads, bins, df.index, start=start)
    grp = pd.DataFrame(df.values, columns=df.columns, index=dtindex)
    return grp.groupby(level=0), dtindex


def grp_to_datetimeindex(grps, bins, dtindex, start=False):
    """Makes a datetimeindex that has for each entry the timestamp
    of the bin beginning or end this entry belongs to.

    Parameters
    ----------
    grps : numpy.array
        group numbers made by np.digitize(data, bins)
    bins : list
        bin start values e.g. [0,11,21] would be two bins one with
        values 0<=x<11 and the second one with 11<=x<21
    dtindex : pandas.DatetimeIndex
        same length as grps, gives the basis datetime for each group
    start : boolean, optional
        if set to True the start of the bin will be the timestamp for each
        observations

    Returns
    -------
    grpdt : pd.DatetimeIndex
        Datetimeindex where every date is the end of the bin the datetime
        ind the input dtindex belongs to
    """

    dtlist = []
    offset = 1
    index_offset = 0
    # select previous bin and do not subtract a day if start is set to True
    if start:
        offset = 0
        index_offset = -1

    for i, el in enumerate(dtindex):
        _, max_day_month = calendar.monthrange(el.year, el.month)

        dtlist.append(date(el.year, el.month, min([bins[grps[i] + index_offset]
                                                   - offset, max_day_month])))

    return pd.DatetimeIndex(dtlist)


def grouped_dates_between(start_date, end_date, bins=[1, 11, 21, 32], start=False):
    """
    Between a start and end date give all dates that represent a bin
    See test for example.

    Parameters
    ----------
    start_date: date
        start date
    end_date: date
        end date
    bins: list, optional
        bin start values as days in a month e.g. [0,11,21] would be two bins one with
        values 0<=x<11 and the second one with 11<=x<21
    start: boolean, optional
        if True the start of the bins is the representative date

    Returns
    -------
    tstamps : list of datetimes
        list of representative dates between start and end date
    """

    daily = pd.date_range(start_date, end_date, freq='D')
    fake_data = pd.DataFrame(np.arange(len(daily)), index=daily)
    grp, dtindex = group_by_day_bin(fake_data, bins=bins, start=start)
    tstamps = grp.sum().index.to_pydatetime().tolist()

    return tstamps


@dataclass
class YearlessDatetime:
    """
    Container class to store Datetime information without a year. This is
    used to group data when the year is not relevant (e.g. seasonal analysis).
    Only down to second. Used by
    :class:`pytesmo.validation_framework.metric_calculators_adapters.TsDistributor`
    """
    month: int

    day: int = 1
    hour: int = 0
    minute: int = 0
    second: int = 0

    @property
    def __ly(self):
        return 2400  # arbitrary leap year

    def __ge__(self, other: 'YearlessDatetime'):
        return self.to_datetime(self.__ly) >= other.to_datetime(self.__ly)

    def __le__(self, other: 'YearlessDatetime'):
        return self.to_datetime(self.__ly) <= other.to_datetime(self.__ly)

    def __lt__(self, other: 'YearlessDatetime'):
        return self.to_datetime(self.__ly) < other.to_datetime(self.__ly)

    def __gt__(self, other: 'YearlessDatetime'):
        return self.to_datetime(self.__ly) > other.to_datetime(self.__ly)

    def __repr__(self):
        return f"****-{self.month:02}-{self.day:02}" \
               f"T{self.hour:02}:{self.minute:02}:{self.second:02}"

    @property
    def doy(self) -> int:
        """
        Get day of year for this date. Assume leap year!
        i.e.: 1=Jan.1st, 366=Dec.31st, 60=Feb.29th.
        """
        return doy(self.month, self.day, year=None)

    @classmethod
    def from_datetime(cls, dt: datetime):
        """
        Omit year from passed datetime to create generic datetime.
        """
        return cls(dt.month, dt.day, dt.hour, dt.minute, dt.second)

    def to_datetime(self, years: Optional[Union[Tuple[int, ...], int]]) \
            -> Union[datetime, List, None]:
        """
        Convert generic datetime to datetime with year.
        Feb 29th for non-leap-years will return None
        """
        dt = []

        for year in np.atleast_1d(years):
            if not calendar.isleap(year) and self.doy == 60.:
                continue
            else:
                d = datetime(year, self.month, self.day, self.hour,
                             self.minute, self.second)
            dt.append(d)

        if len(dt) == 1:
            return dt[0]
        elif len(dt) == 0:
            return None
        else:
            return dt


class TsDistributor:

    def __init__(self,
                 dates=None,
                 date_ranges=None,
                 yearless_dates=None,
                 yearless_date_ranges=None):
        """
        Build a data distibutor from individual dates, date ranges, generic
        dates (without specific year) and generic date ranges.

        Components:
            - individual datetime objects for distinct dates
            - generic datetime objects for dates without specific a year
            - date range / datetime tuple
                i.e. ALL datetimes between the 2 passed dates (start, end)
                the start date must be earlier than the end date
            - generic date range / generic datetime tuple
                i.e. ALL datetimes between 2 generic dates (for any year)

        Parameters
        ----------
        dates : Tuple[datetime, ...] or Tuple[str, ...] or pd.DatetimeIndex
            Individual dates (that also have a year assigned).
        date_ranges: Tuple[Tuple[datetime, datetime], ...]
            A list of date ranges, consisting of a start and end date for each
            range. The start date must be earlier in time than the end date.
        yearless_dates: Tuple[YearlessDatetime,...] or Tuple[datetime...]
            A list of generic dates (that apply to any year).
            Can be passed as a list of
             - YearlessDatetime objects
                    e.g. YearlessDatetime(5,31,12,1,10), ie. May 31st 12:01:10
             - pydatetime objects (years will be ignored, duplicates dropped)
        yearless_date_ranges: [Tuple[YearlessDatetime, YearlessDatetime], ...]
            A list of generic date ranges (that apply to any year).
        """

        self.dates = dates
        self.date_ranges = date_ranges
        self.yearless_dates = yearless_dates
        self.yearless_date_ranges = yearless_date_ranges

    def __repr__(self):
        s = []
        for var in ['dates', 'date_ranges', 'yearless_dates',
                    'yearless_date_ranges']:
            val = getattr(self, var)
            s.append(f"#{var}={len(val) if val is not None else 0}")

        return f"{self.__class__.__name__}({', '.join(s)})"

    def select(self,
               df: Union[pd.DataFrame, pd.Series, pd.DatetimeIndex],
               set_nan=False):
        """
        Select rows from data frame or series with mathing date time indices.

        Parameters
        ----------
        df: pd.DataFrame or pd.Series
            Must have a date time index, which is then filtered based on the
            dates.
        set_nan: bool, optional (default: False)
            Instead of dropping rows that are not selected, set their values to
            nan.


        Returns
        -------
        df: pd.DataFrame or pd.Series
            The filterd input data

        """
        if isinstance(df, pd.DatetimeIndex):
            idx = df
        else:
            idx = df.index

            if not isinstance(idx, pd.DatetimeIndex):
                raise ValueError(f"Expected a DatetimeIndex, "
                                 f"but got {type(df.index)}.")

        mask = self.filter(idx)

        if set_nan:
            df[~mask] = np.nan
            return df
        else:
            return df[mask]

    def filter(self, idx: pd.DatetimeIndex):
        """
        Filter datetime index for a TimeSeriesDistributionSet

        Parameters
        ----------
        idx: pd.DatetimeIndex
            Datetime index to split using the set

        Returns
        -------
        idx_filtered: pd.DatetimeIndex
            Filtered Index that contains dates for the set
        """

        mask = pd.DataFrame(index=idx.copy())

        if self.dates is not None:
            _idx_dates = idx.intersection(pd.DatetimeIndex(self.dates))
            mask['dates'] = False
            mask.loc[_idx_dates, 'dates'] = True

        if self.date_ranges is not None:
            for i, drange in enumerate(self.date_ranges):
                start, end = drange[0], drange[1]
                if start > end:
                    start, end = end, start
                mask[f"range{i}"] = (idx >= start) & (idx <= end)

        if self.yearless_dates is not None:
            arrs = np.array([])
            for d in self.yearless_dates:
                dts = d.to_datetime(np.unique(idx.year))
                if dts is None:
                    continue
                else:
                    arrs = np.append(arrs, dts)
            _idx_dates = idx.intersection(pd.DatetimeIndex(arrs))
            mask['gen_dates'] = False
            mask.loc[_idx_dates, 'gen_dates'] = True

    # avoid loop like:
    # cond = ["__index_month == {}".format(m) for m in months]
    # selection = dat.query(" | ".join(cond)).index

        if self.yearless_date_ranges is not None:
            cols = {}

            for i, gdrange in enumerate(self.yearless_date_ranges):
                for y in np.unique(idx.year):
                    start = gdrange[0]

                    if not calendar.isleap(y) and (gdrange[0].doy == 60):
                        start = YearlessDatetime(3, 1)

                    start_dt = start.to_datetime(years=y)

                    end = gdrange[1]

                    if end < start:
                        y += 1

                    if (not calendar.isleap(y)) and (end.doy == 60):
                        end = YearlessDatetime(2, 28, 23, 59, 59)

                    end_dt = end.to_datetime(years=y)

                    cols[f"gen_range{y}-{i}"] = (idx >= start_dt) & (
                        idx <= end_dt)

            mask = pd.concat(
                [mask, pd.DataFrame(index=mask.index, data=cols)],
                axis=1)

        return mask.any(axis=1, bool_only=True)
