# Copyright (c) 2021, Vienna University of Technology (TU Wien), Department
# of Geodesy and Geoinformation (GEO).
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#   * Neither the name of the Vienna University of Technology, Department
#     of Geodesy and Geoinformation nor the names of its contributors may
#     be used to endorse or promote products derived from this software
#     without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL VIENNA UNIVERSITY OF TECHNOLOGY,
# DEPARTMENT OF GEODESY AND GEOINFORMATION BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import warnings
from itertools import groupby

import pandas as pd
from scipy.stats import linregress

from pygeobase.object_base import TS
from pytesmo.temporal_matching import temporal_collocation as tcoll


class MixinReadTs():
    """Mixin class to provide the reading function in DataAverager and DataManager"""

    def read_ds(self, name, *args):
        """
        Function to read and prepare a datasets.

        Calls read_ts of the dataset.

        Takes either 1 (gpi) or 2 (lon, lat) arguments.

        Parameters
        ----------
        name : string
            Name of the other dataset.
        gpi : int
            Grid point index
        lon : float
            Longitude of point
        lat : float
            Latitude of point

        Returns
        -------
        data_df : pandas.DataFrame or None
            Data DataFrame.

        """
        ds = self.datasets[name]
        args = list(args)
        args.extend(ds['args'])

        try:
            func = getattr(ds['class'], self.read_ts_names[name])
            data_df = func(*args, **ds['kwargs'])
            if type(data_df) is TS or issubclass(type(data_df), TS):
                data_df = data_df.data
        except IOError:
            warnings.warn(
                "IOError while reading dataset {} with args {:}".format(name,
                                                                        args))
            return None
        except RuntimeError as e:
            if e.args[0] == "No such file or directory":
                warnings.warn(
                    "IOError while reading dataset {} with args {:}".format(name,
                                                                            args))
                return None
            else:
                raise e

        if len(data_df) == 0:
            warnings.warn("No data for dataset {}".format(name))
            return None

        if isinstance(data_df, pd.DataFrame) == False:
            warnings.warn("Data is not a DataFrame {:}".format(args))
            return None

        if self.period is not None:
            # here we use the isoformat since pandas slice behavior is
            # different when using datetime objects.
            data_df = data_df[
                      self.period[0].isoformat():self.period[1].isoformat()]

        if len(data_df) == 0:
            warnings.warn("No data for dataset {} with arguments {:}".format(name,
                                                                             args))
            return None

        else:
            return data_df


class DataAverager(MixinReadTs):
    """
    Class to handle multiple measurements falling under the same reference gridpoint. The goal is to include
    here all identified upscaling methods to provide an estimate at the reference footprint scale.

    Implemented methods:

        * time-stability filtering
        * simple averaging

    Parameters
    ----------
    ref_class: pygeogrids Grid obj.
        Class containing the method read_ts for reading the data of the reference
    others_class: dict
        Dict of shape 'other_name': pygeogrids Grid obj. for the other dataset
    geo_subset: tuple, optional. Default is None.
        Information on the grographic subset for data averaging -> (latmin, latmax, lonmin, lonmax)
    manager_parms: dict
        Dict of DataManager attributes
    """

    def __init__(
            self,
            ref_class,
            others_class,
            geo_subset,
            manager_parms,
    ):
        self.ref_class = ref_class
        self.others_class = others_class
        self.geo_subset = geo_subset
        # attributes used by the Mixin class:
        self.datasets = manager_parms["datasets"]
        self.period = manager_parms["period"]
        self.read_ts_names = manager_parms["read_ts_names"]

    @property
    def lut(self) -> dict:
        """Get a lookup table that combines the points falling under the same reference pixel"""
        lut = {}
        for other_name, other_class in self.others_class.items():
            try:
                grid = other_class.grid
            except AttributeError:
                # when filters are applied, this will be an AdvancedMaskingAdapter
                grid = other_class.cls.cls.grid
            # subsetting shouldn't be necessary with ismn reader, but for satellites
            other_points = grid.get_bbox_grid_points(
                *self.geo_subset,
                both=True,
            )
            other_lut = {}
            # iterate from the side of the non-reference
            for gpi, lat, lon in zip(other_points[0], other_points[1], other_points[2]):
                # list all non-ref points under the same ref gpi
                try:
                    ref_gpi = self.ref_class.grid.find_nearest_gpi(lon, lat)[0]
                except AttributeError:
                    ref_gpi = self.ref_class.cls.cls.grid.find_nearest_gpi(lon, lat)[0]
                if ref_gpi in other_lut.keys():
                    other_lut[ref_gpi].append((gpi, lon, lat))
                else:
                    other_lut[ref_gpi] = [(gpi, lon, lat)]
            # add to dictionary even when empty
            lut[other_name] = other_lut

        return lut

    def get_timeseries(
            self,
            points,
            other_name,
    ) -> pd.DataFrame:
        """
        Get the timeseries for given points info and return them in a list.

        Parameters
        ----------
        points: list of tuples
            list of tuples of (gpi, lon, lat)
        other_name: str
            Name of the dataset which the points belong to

        Returns
        -------
        dss: list
            list of dataframes of the reference timeseries
        """
        dss = []
        for gpi, lon, lat in points:
            ds = self.read_ds(other_name, gpi)
            dss.append(ds)

        return dss

    @staticmethod
    def temp_match(
            to_match,
            method='rescale',
            hours=6,
            **kwargs
    ) -> pd.DataFrame:
        """
        Temporal match by:
            * temporal matching to the longest timeseries
            * taking the common values only

        Parameters
        ----------
        to_match: list
            list of dataframes to match
        method: str
            matching method. Either 'common' or 'rescale'
        hours: int
            window to perform the temporal matching

        Returns
        -------
        matched: pd.DataFrame
            dataframe with temporally matched timeseries
        """
        if method == 'common':
            matched = pd.concat(to_match, axis=1, join="inner").dropna(how="any")

        elif method == 'rescale':
            # get time series with most points
            for n, df in enumerate(to_match):
                if df is None:
                    continue
                points = len(df.dropna())
                if n == 0:
                    ref = df
                if len(ref.dropna()) >= points:
                    continue
                else:
                    ref = df

            to_match = pd.concat(to_match, axis=1)
            matched = tcoll(
                ref,
                to_match,
                pd.Timedelta(hours, "H"),
                dropna=True
            ).dropna(how="any")

            # todo: handle cases with no match with warning

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
        Uses time stability concepts to filter point-measurements (pms). Determines wether the upscaled measurement
        based on a simple average of all the pms is in sufficient agreement with each pm, and if not eliminates pm
        from the pool.

        Parameters
        ----------
        df: pd.DataFrame
            temporally matched DataFrame
        r_min: float
            lower threshold for correlation (Pearson) between upscaled and pm
        see_max: float
            upper threshold for standard error of estimate
        min_n: int
            minimum number of pms to perform the filtering

        Returns
        -------
        filtered: pd.DataFrame
            filtered input
        """
        if len(df.columns) < min_n:
            return df

        # get a trivial (average) upscale estimate
        estimate = df.mean(axis=1)

        filter_out = []
        for n, pm in enumerate(df):
            pm_values = df[pm]
            regr = linregress(
                estimate,
                pm_values
            )
            if regr.rvalue < r_min or regr.intercept_stderr > see_max:
                filter_out.append(pm)

        filtered = df.drop(filter_out)
        # todo: This could be done iteratively during the run to calibrate the thresholds
        if len(filtered.columns) < 2:
            warnings.warn(
                "The filtering options are too strict. Returning entire dataframe ..."
            )
            return df

        return filtered

    def upscale(self, df, method='average', **kwargs) -> pd.Series:
        """
        Handle the column names and return the upscaled Dataframe with the specified method

        **New upscaling methods can be specified here in the lut**

        Parameters
        ----------
        df: pd.DataFrame
            Dataframe of values to upscale using method
        method: str
            averaging method
        kwargs: keyword arguments
            Arguments for some upscaling functions

        Returns
        -------
        upscaled: pandas.DataFrame
            dataframe with "upscaled" column
        """
        up_function = {
            "average": self.faverage,
        }
        f = up_function[method]

        return f(df, **kwargs)

    @staticmethod
    def faverage(df) -> pd.Series:
        """Simple average of each column in the dataframe"""
        # workaround to avoid changes in place
        out = df.mean(axis=1)

        return out

    def wrapper(
            self,
            gpi,
            other_name,
            tmatching="rescale",
            up_method="average",
            tstability=False,
            **kwargs
    ) -> pd.DataFrame:
        """
        Find the upscale estimate with given method, for a certain reference gpi

        Parameters
        ----------
        gpi: int
            gpi value of the reference point
        other_name: str
            name of the non-reference dataset to be upscaled
        tmatching: str, default is "rescale"
            method to use for temporal matching. 'rescale' uses the pytesmo.temporal_matching methods, 'common' takes
            only measurements at common times between all the pms
        up_method: str
            method to use for upscaling:
                * 'average' takes the simple mean of all timeseries
        tstability: bool, default is False
            if True, the values are filtered using the time stability concept
        kwargs: keyword arguments
            argumjents for the temporal window or time stability thresholds

        Returns
        -------
        upscaled: pd.DataFrame
            upscaled time series
        """
        other_lut = self.lut[other_name]
        # check that there are points for specific reference gpi
        if not gpi in other_lut.keys():
            warnings.warn(
                "The reference gpi {} has no points to average from {}".format(gpi, other_name)
            )
            return None
        else:
            other_points = other_lut[gpi]

        # read non-reference points and filter out None
        tss = self.get_timeseries(
            points=other_points,
            other_name=other_name
        )
        tss = [df for df in tss if not df is None]

        # handle situation with single timeseries or all None
        if len(tss) <= 1:
            if not tss:
                return None
            return tss[0]

        # here we collect only the variable columns; flags are irrelevant at this point and can be dropped
        target_column = self.datasets[other_name]["columns"][0]
        to_match = []
        for n, point_df in enumerate(tss):
            point_ts = point_df[target_column]
            to_match.append(
                point_ts.to_frame(name=target_column + "_{}".format(n))  # avoid name clashing
            )

        # temporal match and time stability filtering
        tss = self.temp_match(to_match, method=tmatching, **kwargs)
        if tstability:
            tss = self.tstability_filter(tss, **kwargs)

        # perform upscaling and return correct name
        upscaled = self.upscale(
            tss,
            method=up_method
        ).to_frame(target_column)

        return upscaled
