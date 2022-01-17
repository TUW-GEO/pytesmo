# Copyright (c) 2020, TU Wien, Department of Geodesy and Geoinformation
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of the TU Wien, Department of Geodesy and
#      Geoinformation nor the names of its contributors may be used to endorse
#      or promote products derived from this software without specific prior
#      written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL VIENNA UNIVERSITY OF TECHNOLOGY,
# DEPARTMENT OF GEODESY AND GEOINFORMATION BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
Metric Calculator Adapters change how metrics are calculated by calling
the `calc_metric` function of the adapted calculator instead of the unadapted
version.
"""
import pandas as pd

from pytesmo.validation_framework.metric_calculators import (
    PairwiseIntercomparisonMetrics, TripleCollocationMetrics)
import warnings
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Union, List, Tuple, Optional
from cadati.conv_doy import doy, days_past
import calendar


def days_in_month(month: int) -> int:
    """
    Get number of days in this month (in a LEAP YEAR)
    """
    return days_past[month] - days_past[month - 1]


@dataclass
class GenericDatetime:
    """
    Datetime without a year. Only down to second.
    """
    month: int

    day: int = 1
    hour: int = 0
    minute: int = 0
    second: int = 0

    @property
    def __ly(self):
        return 2400  # arbitrary leap year

    def __ge__(self, other: 'GenericDatetime'):
        return self.to_datetime(self.__ly) >= other.to_datetime(self.__ly)

    def __le__(self, other: 'GenericDatetime'):
        return self.to_datetime(self.__ly) <= other.to_datetime(self.__ly)

    def __lt__(self, other: 'GenericDatetime'):
        return self.to_datetime(self.__ly) < other.to_datetime(self.__ly)

    def __gt__(self, other: 'GenericDatetime'):
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
                 generic_dates=None,
                 generic_date_ranges=None):
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
        generic_dates: Tuple[GenericDateTime,...] or Tuple[datetime...]
            A list of generic dates (that apply to any year).
            Can be passed as a list of
             - GenericDateTime objects
                    e.g. GenericDateTime(5,31,12,1,10), ie. May 31st 12:01:10
             - pydatetime objects (years will be ignored, duplicates dropped)
        generic_date_ranges: [Tuple[GenericDateTime, GenericDateTime], ...]
            A list of generic date ranges (that apply to any year).
        """

        self.dates = dates
        self.date_ranges = date_ranges
        self.generic_dates = generic_dates
        self.generic_date_ranges = generic_date_ranges

    def __repr__(self):
        s = []
        for var in [
                'dates', 'date_ranges', 'generic_dates', 'generic_date_ranges'
        ]:
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

        if self.generic_dates is not None:
            arrs = np.array([])
            for d in self.generic_dates:
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

        if self.generic_date_ranges is not None:
            for i, gdrange in enumerate(self.generic_date_ranges):
                for y in np.unique(idx.year):

                    if not calendar.isleap(y) and (gdrange[0].doy == 60):
                        start = GenericDatetime(3, 1)
                    else:
                        start = gdrange[0]

                    if (not calendar.isleap(y)) and (gdrange[1].doy == 60):
                        end = GenericDatetime(2, 28, 23, 59, 59)
                    else:
                        end = gdrange[1]

                    start_dt = start.to_datetime(years=y)

                    if end < start:
                        end_dt = end.to_datetime(years=y + 1)
                    else:
                        end_dt = end.to_datetime(years=y)

                    mask[f"gen_range{y}-{i}"] = (idx >= start_dt) & (
                        idx <= end_dt)

        return mask.any(axis=1, bool_only=True)


class SubsetsMetricsAdapter:
    """
    Adapt MetricCalculators to calculate metrics for groups of temporal
    subsets (also across multiple years).
    """

    _supported_metric_calculators = (
        PairwiseIntercomparisonMetrics,
        TripleCollocationMetrics,
    )

    def __init__(self, calculator, subsets):
        """
        Add functionality to a metric calculator to calculate validation
        metrics for subsets of certain datetimes in a time series
        (e.g. seasonal).

        Parameters
        ----------
        calculator : PairwiseIntercomparisonMetrics or TripleCollocationMetrics
            A metric calculator to adapt. Preferably an instance of a metric
             calculator listed in `_supported_metric_calculators`
        subsets : dict[str, TsDistributor], optional (default: None)
            Define subsets of data. With group names as key and a
            data distributor as values.
        """
        if not isinstance(calculator, self._supported_metric_calculators):
            warnings.warn(f"Adapting {calculator.__class__} is not supported.")

        self.cls = calculator

        self.subsets = subsets

        # metadata metrics and lon, lat, gpi are excluded from applying
        # seasonally
        self.non_seas_metrics = ["gpi", "lon", "lat"]
        if self.cls.metadata_template is not None:
            self.non_seas_metrics += list(self.cls.metadata_template.keys())

        all_metrics = calculator.result_template
        subset_metrics = {}

        # for each subset create a copy of the metric template
        for name in subsets.keys():
            for k, v in all_metrics.items():
                if k in self.non_seas_metrics:
                    subset_metrics[k] = np.array(v)
                else:
                    subset_metrics[(name, k)] = np.array(v)

        self.result_template = subset_metrics

    def calc_metrics(self, data, gpi_info):
        """
        Calculates the desired statistics, for each set that was defined.

        Parameters
        ----------
        data : pandas.DataFrame
            with 2 columns, the first column is the reference dataset
            named 'ref'
            the second column the dataset to compare against named 'other'
        gpi_info : tuple
            Grid point info (i.e. gpi, lon, lat)
        """
        dataset = self.result_template.copy()

        for setname, distr in self.subsets.items():
            df = distr.select(data)
            ds = self.cls.calc_metrics(df, gpi_info=gpi_info)
            for metric, res in ds.items():
                if metric in self.non_seas_metrics:
                    k = f"{metric}"
                else:
                    k = (f"{setname}", *np.atleast_1d(metric))
                dataset[k] = res

        return dataset


class MonthsMetricsAdapter(SubsetsMetricsAdapter):
    """
    Adapt MetricCalculators to calculate metrics for groups across months
    """

    def __init__(self, calculator, sets=None):
        """
        Add functionality to a metric calculator to calculate validation
        metrics for subsets of certain months in a time series (e.g. seasonal).

        Parameters
        ----------
        calculator : PairwiseIntercomparisonMetrics or TripleCollocationMetrics
            A metric calculator to adapt. Preferably an instance of a metric
             calculator listed in `_supported_metric_calculators`
        sets : dict, optional (default: None)
            Define groups of data. With group names as key and a list of
            months (1-12) that belong to the group as values.

            e.g. {'Group1': [4,5,6,7,8,9], 'Group2': [10,11,12,1,2,3]} will
            split the data used by the metric calculator into 2 groups.
            One using only observations made between April and September,
            and one using observations from the rest of the year.

            The name will be used in the results to distinguish between the
            same metrics for different groups:
            e.g. ('Group1', 'BIAS'): ..., ('Group2', 'BIAS'): ..., etc.

            The default groups are based on 4 seasons plus one group that uses
            all data (as the unadapted metric calculator would do):
            {'DJF': [12,1,2], 'MAM': [3,4,5], 'JJA': [6, 7, 8],
             'SON': [9, 10, 11], 'ALL': list(range(1, 13))}
        """
        if sets is None:
            sets = {
                'DJF': [12, 1, 2],
                'MAM': [3, 4, 5],
                'JJA': [6, 7, 8],
                'SON': [9, 10, 11],
                'ALL': list(range(1, 13)),
            }

        for name, months in sets.items():
            distr = TsDistributor(generic_date_ranges=[(
                GenericDatetime(m, 1, 0, 0, 0),
                GenericDatetime(m, days_in_month(m), 23, 59, 59))
                                                       for m in months])
            sets[name] = distr

        super().__init__(calculator, subsets=sets)


if __name__ == '__main__':
    dt = GenericDatetime(2, 29).to_datetime(2001)
