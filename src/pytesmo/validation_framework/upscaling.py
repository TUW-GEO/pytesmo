# Copyright (c) 2021, TU Wien, Department of Geodesy and Geoinformation
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


import warnings
from typing import Union

import pandas as pd
from scipy.stats import linregress

from pygeobase.object_base import TS
from pytesmo.temporal_matching import combined_temporal_collocation


class MixinReadTs:
    """Mixin class to provide the reading function in DataAverager
    and DataManager"""

    def read_ds(self, name, *args):
        """
        Function to read and prepare a datasets.

        Calls read_ts of the dataset.

        Takes either 1 (gpi) or 2 (lon, lat) arguments.

        Parameters
        ----------
        name : string
            Name of the other dataset.
        args : either gpi or (lon, lat)
            * gpi (int): Grid point index
            * lon (float): Longitude of point
            * lat(float): Latitude of point

        Returns
        -------
        data_df : pandas.DataFrame or None
            Data DataFrame.

        """
        ds = self.datasets[name]
        args = list(args)
        args.extend(ds["args"])

        try:
            func = getattr(ds["class"], self.read_ts_names[name])
            data_df = func(*args, **ds["kwargs"])
            if type(data_df) is TS or issubclass(type(data_df), TS):
                data_df = data_df.data
        except IOError:
            warnings.warn(
                "IOError while reading dataset {} with args {:}".format(
                    name, args
                )
            )
            return None
        except RuntimeError as e:
            if e.args[0] == "No such file or directory":
                warnings.warn(
                    "IOError while reading dataset {} with args {:}".format(
                        name, args
                    )
                )
                return None
            else:
                raise e

        if len(data_df) == 0:
            warnings.warn("No data for dataset {}".format(name))
            return None

        if not isinstance(data_df, pd.DataFrame):
            warnings.warn("Data is not a DataFrame {:}".format(args))
            return None

        if self.period is not None:
            # here we use the isoformat since pandas slice behavior is
            # different when using datetime objects.
            data_df = data_df[
                self.period[0].isoformat(): self.period[1].isoformat()
            ]

        if len(data_df) == 0:
            warnings.warn(
                "No data for dataset {} with arguments {:}".format(name, args)
            )
            return None

        else:
            return data_df


class Upscaling(MixinReadTs):
    """
    This class provides methods to combine the measurements of validation
    datasets (others) that fall under the same gridpoint of the dataset being
    validated (reference).

    The goal is to include here all identified upscaling methods to provide
    an estimate at the reference footprint scale.

    Implemented methods:

        * time-stability filtering
        * simple averaging

    Parameters
    ----------
    ref_class : <reader object> of the reference
        Class containing the method read_ts for reading the data of
        the reference
    others_class : dict
        Dict of shape {'other_name': <reader object>} for the other dataset
    upscaling_lut : dict
        Dict of shape {'other_name':{ref gpi: [other gpis]}}
    manager_parms : dict
        Dict of DataManager attributes
    """

    def __init__(
        self,
        ref_class,
        others_class,
        upscaling_lut,
        manager_parms,
    ):
        self.ref_class = ref_class
        self.others_class = others_class
        self.lut = upscaling_lut
        # attributes used by the Mixin class:
        self.datasets = manager_parms["datasets"]
        self.period = manager_parms["period"]
        self.read_ts_names = manager_parms["read_ts_names"]

    def _read(
        self,
        points,
        other_name,
    ) -> list:
        """
        Get the timeseries for given points info and return them in a list.

        Parameters
        ----------
        points : list of tuples
            list of tuples of (gpi, lon, lat)
        other_name : str
            Name of the dataset which the points belong to

        Returns
        -------
        dss : list
            list of dataframes of the reference timeseries
        """
        dss = []
        for gpi, lon, lat in points:
            # todo: check attributes in loop
            ds = self.read_ds(other_name, gpi)
            dss.append(ds)

        return dss

    @staticmethod
    def temporal_match(
        to_match, hours=6, drop_missing=False, **kwargs
    ) -> pd.DataFrame:
        """
        Temporal match to the longest timeseries

        Parameters
        ----------
        to_match : list
            list of dataframes to match
        hours : int
            window to perform the temporal matching
        drop_missing : bool, optional. Default is False.
            If true, only time steps when all points have measurements
            are kept

        Returns
        -------
        matched: pd.DataFrame
            dataframe with temporally matched timeseries
        """
        # get time series with most points
        ref = to_match[0]
        for n, df in enumerate(to_match):
            if df is None:
                continue
            points = int(df.count().iloc[0])
            if int(ref.count().iloc[0]) >= points:
                continue
            else:
                ref = df

        combined_dropna = False
        if drop_missing:
            combined_dropna = "any"

        matched = combined_temporal_collocation(
            ref,
            to_match,
            pd.Timedelta(hours, "h"),
            combined_dropna=combined_dropna,
            checkna=True,
        )
        matched.dropna(axis="columns", how="all", inplace=True)

        return matched

    @staticmethod
    def tstability_filter(
        df,
        r_min=0.6,
        see_max=0.05,
        min_n=4,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Uses time stability concepts to filter point-measurements (pms).
        Determines whether the upscaled measurement based on a simple average
        of all the pms is in sufficient agreement with each pm, and if not
        eliminates pm from the pool.

        Thresholds are based on:
        Wagner W, Pathe C, Doubkova M, Sabel D, Bartsch A, Hasenauer S,
        Blöschl G, Scipal K, Martínez-Fernández J, Löw A.
        Temporal Stability of Soil Moisture and Radar Backscatter Observed
        by the Advanced Synthetic Aperture Radar (ASAR). Sensors. 2008;
        8(2):1174-1197. https://doi.org/10.3390/s80201174

        Parameters
        ----------
        df : pd.DataFrame
            temporally matched DataFrame
        r_min : float
            lower threshold for correlation (Pearson) between upscaled and pm
        see_max : float
            upper threshold for standard error of estimate
        min_n : int
            minimum number of pms to perform the filtering

        Returns
        -------
        filtered : pd.DataFrame
            filtered input
        """
        if len(df.columns) < min_n:
            return df

        # get a trivial (average) upscale estimate
        estimate = df.mean(axis=1)

        filter_out = []
        for n, pm in enumerate(df):
            pm_values = df[pm]
            regr = linregress(estimate, pm_values)
            if regr.rvalue < r_min or regr.stderr > see_max:
                filter_out.append(pm)

        filtered = df.drop(filter_out, axis="columns")

        return filtered

    def upscale(self, df, method="average", **kwargs) -> pd.Series:
        """
        Handle the column names and return the upscaled Dataframe with the
        specified method.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe of values to upscale using method
        method : str
            averaging method
        kwargs : keyword arguments
            Arguments for some upscaling functions

        Returns
        -------
        upscaled : pandas.DataFrame
            dataframe with "upscaled" column
        """
        # New upscaling methods can be specified here in the lookup table
        up_function = {
            "average": self._average,
        }
        if method not in up_function.keys():
            raise KeyError(
                "The selected method {} is not implemented in the "
                "upscaling options".format(
                    method
                )
            )
        f = up_function[method]

        return f(df, **kwargs)

    @staticmethod
    def _average(df, **kwargs) -> pd.Series:
        """Simple average of each column in the dataframe"""
        out = df.mean(axis=1)

        return out

    def get_upscaled_ts(
        self,
        gpi,
        other_name,
        upscaling_method="average",
        temporal_stability=False,
        **kwargs,
    ) -> Union[None, pd.DataFrame]:
        """
        Find the upscale estimate timeseries with given method, for a certain
        reference gpi

        Parameters
        ----------
        gpi : int
            gpi value of the reference point
        other_name : str
            name of the non-reference dataset to be upscaled
        upscaling_method : str
            method to use for upscaling:
                * 'average' takes the simple mean of all timeseries
        temporal_stability : bool, default is False
            if True, the values are filtered using the time stability concept
        kwargs : keyword arguments
            arguments for the temporal window or time stability thresholds

        Returns
        -------
        upscaled : pd.DataFrame or None
            upscaled time series; if there are no points under the specific
            gpi, None is returned
        """
        other_lut = self.lut[other_name]
        # check that there are points for specific reference gpi
        if gpi not in other_lut.keys():
            return None
        else:
            other_points = other_lut[gpi]

        # read non-reference points and filter out Nones
        tss = self._read(points=other_points, other_name=other_name)
        tss = [df for df in tss if df is not None]

        # handle situation with single timeseries or all None
        if len(tss) <= 1:
            if not tss:
                return None
            return tss[0]

        # here we collect only the variable columns; flags are irrelevant at
        # this point and can be dropped
        target_column = self.datasets[other_name]["columns"][0]
        to_match = []
        for n, point_df in enumerate(tss):
            point_ts = point_df[target_column]
            to_match.append(
                point_ts.to_frame(
                    name=target_column + "_{}".format(n)
                )  # avoid name clashing
            )

        tss = self.temporal_match(to_match, **kwargs)
        if temporal_stability:
            tss = self.tstability_filter(tss, **kwargs)

        # perform upscaling and return correct name
        upscaled = self.upscale(tss, method=upscaling_method).to_frame(
            target_column
        )

        return upscaled
