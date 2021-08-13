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

from pytesmo.validation_framework.metric_calculators import (
    PairwiseIntercomparisonMetrics,
    TripleCollocationMetrics
)
import warnings
import numpy as np


class MonthsMetricsAdapter(object):
    """
    Adapt MetricCalculators to calculate metrics for groups across months
    """

    _supported = (
        PairwiseIntercomparisonMetrics,
        TripleCollocationMetrics,
    )

    def __init__(self, calculator, sets=None):
        """
        Add functionality to a metric calculator to calculate validation
        metrics for subsets of certain months in a time series (e.g. seasonal).

        Parameters
        ----------
        calculator : MetadataMetrics or any child of it (not Rolling metrics)
        sets : dict, optional (default: None)
            A dictionary consisting of a set name (which is added to the metric
            name as a suffix) and the list of months that belong to that set.
            If None is passed, we use 4 (seasonal) sets named after the fist
            letter of each month used.
        """
        if not isinstance(calculator, self._supported):
            warnings.warn(f"Adapting {calculator.__class__} is not supported.")
        self.cls = calculator
        if sets is None:
            sets = {
                "DJF": [12, 1, 2],
                "MAM": [3, 4, 5],
                "JJA": [6, 7, 8],
                "SON": [9, 10, 11],
                "ALL": list(range(1, 13)),
            }

        self.sets = sets

        # metadata metrics and lon, lat, gpi are excluded from applying
        # seasonally
        self.non_seas_metrics = ["gpi", "lon", "lat"]
        if self.cls.metadata_template is not None:
            self.non_seas_metrics += list(self.cls.metadata_template.keys())

        all_metrics = calculator.result_template
        subset_metrics = {}

        # for each subset create a copy of the metric template
        for name in sets.keys():
            for k, v in all_metrics.items():
                if k in self.non_seas_metrics:
                    subset_metrics[f"{k}"] = v
                else:
                    subset_metrics[(f"{name}", f"{k}")] = v

        self.result_template = subset_metrics

    @staticmethod
    def filter_months(df, months, dropna=False):
        """
        Select only entries of a time series that are within certain month(s)

        Parameters
        ----------
        df : pd.DataFrame
            Time series (index.month must exist) that is filtered
        months : list
            Months for which data is kept, e.g. [12,1,2] to keep data for
            winter
        dropna : bool, optional (default: False)
            Drop lines for months that are not to be kept, if this is false,
            the original index is not changed, but filtered values are replaced
            with nan.

        Returns
        -------
        df_filtered : pd.DataFrame
            The filtered series
        """
        dat = df.copy(True)
        dat["__index_month"] = dat.index.month
        cond = ["__index_month == {}".format(m) for m in months]
        selection = dat.query(" | ".join(cond)).index
        dat.drop("__index_month", axis=1, inplace=True)

        if dropna:
            return dat.loc[selection]
        else:
            dat.loc[dat.index.difference(selection)] = np.nan
            return dat

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

        for setname, months in self.sets.items():
            df = self.filter_months(data, months=months, dropna=True)
            ds = self.cls.calc_metrics(df, gpi_info=gpi_info)
            for metric, res in ds.items():
                if metric in self.non_seas_metrics:
                    k = f"{metric}"
                else:
                    k = (f"{setname}", *np.atleast_1d(metric))
                dataset[k] = res

        return dataset
