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

from pytesmo.time_series.grouping import YearlessDatetime, TsDistributor
from pytesmo.validation_framework.metric_calculators import (
    PairwiseIntercomparisonMetrics, TripleCollocationMetrics)
import warnings
import numpy as np
import pandas as pd
from cadati.conv_doy import days_past


def days_in_month(month: int) -> int:
    """
    Get number of days in this month (in a LEAP YEAR)
    """
    return days_past[month] - days_past[month - 1]


class SubsetsMetricsAdapter:
    """
    Adapt MetricCalculators to calculate metrics for groups of temporal
    subsets (also across multiple years).
    """

    _supported_metric_calculators = (
        PairwiseIntercomparisonMetrics,
        TripleCollocationMetrics,
    )

    def __init__(self, calculator, subsets, group_results='tuple'):
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
        group_results: str, optional (default: 'tuple')
            How to group the results.
            - 'tuple' will group the results by (group, metric)
            - 'join' will join group and metric name with a '|'
        """
        if not isinstance(calculator, self._supported_metric_calculators):
            warnings.warn(f"Adapting {calculator.__class__} is not supported.")

        self.cls = calculator
        self.subsets = subsets
        self.group_results = group_results

        assert group_results in ('tuple', 'join'), \
            f"Unknown group_results: {group_results}"

        # metadata metrics and lon, lat, gpi are excluded from applying
        # seasonally
        self.non_seas_metrics = ["gpi", "lon", "lat"]
        if hasattr(self.cls, 'metadata_template'):
            if self.cls.metadata_template is not None:
                self.non_seas_metrics += list(
                    self.cls.metadata_template.keys())

        all_metrics = calculator.result_template
        subset_metrics = {}

        # for each subset create a copy of the metric template
        for name in subsets.keys():
            for k, v in all_metrics.items():
                subset_metrics[self._genname(name, k)] = np.array(v)

        self.result_template = subset_metrics

    def _genname(self, setname: str, metric: str) -> str or tuple:
        if metric in self.non_seas_metrics:
            k = f"{metric}"
        elif self.group_results == 'tuple':
            k = (f"{setname}", *np.atleast_1d(metric))
        elif self.group_results == 'join':
            k = f"{setname}|{metric}"
        else:
            raise NotImplementedError(
                f"Unknown group_results: {self.group_results}")
        return k

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
            if len(data.index) == 0:
                df = pd.DataFrame()
            else:
                df = distr.select(data)
            ds = self.cls.calc_metrics(df, gpi_info=gpi_info)
            for metric, res in ds.items():
                k = self._genname(setname, metric)
                dataset[k] = res

        return dataset


class MonthsMetricsAdapter(SubsetsMetricsAdapter):
    """
    Adapt MetricCalculators to calculate metrics for groups across months
    """

    def __init__(self, calculator, month_subsets=None, group_results='tuple'):
        """
        Add functionality to a metric calculator to calculate validation
        metrics for subsets of certain months in a time series (e.g. seasonal).

        Parameters
        ----------
        calculator : PairwiseIntercomparisonMetrics or TripleCollocationMetrics
            A metric calculator to adapt. Preferably an instance of a metric
             calculator listed in `_supported_metric_calculators`
        month_subsets : dict, optional (default: None)
            Define groups of data. With group names as key and a list of
            months (1-12) that belong to the group as values.

            e.g. {'Group1': [4,5,6,7,8,9], 'Group2': [10,11,12,1,2,3]} will
            split the data used by the metric calculator into 2 groups.
            One using only observations made between April and September,
            and one using observations from the rest of the year.

            The name will be used in the results to distinguish between the
            same metrics for different groups:
            e.g. ('Group1', 'BIAS'): ..., ('Group2', 'BIAS'): ..., etc. or
            'Group1|BIAS': ..., 'Group2|BIAS': ..., etc.
            denpending on the chosen `group_results` parameter.

            The default groups are based on 4 seasons plus one group that uses
            all data (as the unadapted metric calculator would do):
            {'DJF': [12,1,2], 'MAM': [3,4,5], 'JJA': [6, 7, 8],
             'SON': [9, 10, 11], 'ALL': list(range(1, 13))}
        group_results: str, optional (default: 'tuple')
            How to group the results.
            - 'tuple' will group the results by (group, metric)
            - 'join' will join group and metric name with a '|'
        """
        if month_subsets is None:
            month_subsets = {
                'DJF': [12, 1, 2],
                'MAM': [3, 4, 5],
                'JJA': [6, 7, 8],
                'SON': [9, 10, 11],
                'ALL': list(range(1, 13)),
            }

        for name, months in month_subsets.items():
            distr = TsDistributor(yearless_date_ranges=[(
                YearlessDatetime(m, 1, 0, 0, 0),
                YearlessDatetime(m, days_in_month(m), 23, 59, 59))
                                                       for m in months])
            month_subsets[name] = distr

        super().__init__(calculator, subsets=month_subsets,
                         group_results=group_results)
