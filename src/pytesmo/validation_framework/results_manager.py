# -*- coding: utf-8 -*-

"""
The results manager stores validation results in netcdf format.
"""

from netCDF4 import date2num, num2date, Variable
import numpy as np
import os
import pandas as pd
from pynetcf.base import Dataset
from typing import Union
import warnings


def build_filename(root, key):
    """
    Create savepath/filename that does not exceed 255 characters

    Parameters
    ----------
    root : str
        Directory where the file should be stored
    key : list of tuples
        The keys are joined to create a filename from them. If the length of
        the joined keys is too long we shorten it.

    Returns
    -------
    fname : str
        Full path to the netcdf file to store
    """
    ds_names = []
    for ds in key:
        if isinstance(ds, tuple):
            ds_names.append(".".join(ds))
        else:
            ds_names.append(ds)

    fname = "_with_".join(ds_names)
    ext = "nc"

    if len(os.path.join(root, ".".join([fname, ext]))) > 255:
        ds_names = [str(ds[0]) for ds in key]
        fname = "_with_".join(ds_names)

        if len(os.path.join(root, ".".join([fname, ext]))) > 255:
            fname = "validation"

    return os.path.join(root, ".".join([fname, ext]))


class PointDataResults(Dataset):
    def __init__(self, filename, zlib=True, read_only=False):
        """
        Results manager for validation results as returned by the validation
        framework

        Parameters
        ----------
        filename : str
            Path to the netcdf file to add results to
        zlib : bool, optional (default: True)
            Activate/deactivate file compression
        read_only : bool, optional (default: False)
            Force read only mode when opening the netcdf file
        """

        self.time_unit = "days since 1900-01-01 00:00:00"

        if os.path.exists(filename):
            if read_only:
                mode = "r"
            else:
                mode = "a"
        else:
            mode = "w"

        super(PointDataResults, self).__init__(filename, mode=mode, zlib=zlib)

        if mode == "w":
            # in space (static results)
            self.dataset.createDimension("loc", None)
            # in time (rolling metrics)
            self.dataset.createDimension("obs", None)

            # default variables, along loc dim
            self.write_var(
                "lon",
                dim=("loc",),
                dtype="float",
                attr=dict(
                    long_name="location longitude",
                    standard_name="longitude",
                    units="degrees_east",
                    valid_range=np.array([-180, 180]),
                    axis="X",
                ),
            )

            self.write_var(
                "lat",
                dim=("loc",),
                dtype="float",
                attr=dict(
                    long_name="location latitude",
                    standard_name="latitude",
                    units="degrees_north",
                    valid_range=np.array([-90, 90]),
                    axis="Y",
                ),
            )

            self.write_var(
                "idx",
                dim=("loc",),
                dtype="int",
                attr=dict(
                    long_name="observation point index",
                    standard_name="idx",
                    units="timeseries_id",
                ),
            )

            # indexing var for time series
            self.write_var(
                "_row_size",
                dim=("loc",),
                dtype="int",
                attr=dict(
                    long_name="number of timestamps for this loc",
                    sample_dimension="obs",
                ),
            )

            self.write_var(
                "time",
                dtype="float",
                dim=("obs",),
                attr=dict(
                    long_name="metric time stamp",
                    standard_name="time",
                    units=self.time_unit,
                ),
            )

    def __getitem__(self, key: str) -> Variable:
        return self.dataset[key]

    @property
    def variables(self) -> np.array:
        """Names of all variables in the data set"""
        return self.dataset.variables.keys()

    def _num2date(
        self, nums: np.array, as_pd_idx: bool = True
    ) -> Union[pd.DatetimeIndex, np.array]:
        """Read time stamps and convert them."""
        dates = num2date(
            nums,
            units=self.time_unit,
            calendar="standard",
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=True,
        )

        if as_pd_idx:
            return pd.DatetimeIndex(dates)
        else:
            return dates.astype("datetime64[ns]")

    def _date2num(self, dates: Union[pd.DatetimeIndex, np.array]) -> np.array:
        """Convert datetime index or dates to float"""
        if isinstance(dates, pd.DatetimeIndex):
            dates = dates.to_pydatetime()
        return date2num(dates, units=self.time_unit, calendar="standard")

    def _lonlat2idx(self, lon: float, lat: float) -> int:
        """Find index of location by coordinates"""
        return np.where(
            np.logical_and(self["lon"][:] == lon, self["lat"][:] == lat)
        )[0]

    def _idx2lonlat(self, idx: int) -> tuple:
        """Find coordinates of location by index"""
        return (self["lon"][:][idx], self["lat"][:][idx])

    def _sel_attr(self, attr: dict, name: str, time: bool = False) -> dict:
        """Select attributes for variable"""
        if attr is None:
            attr = {}
        if name in attr.keys():
            var_attr = attr[name]
        else:
            var_attr = {}
        if "coordinates" not in var_attr.keys():
            if time:
                var_attr["coordinates"] = "idx lat lon time"
            else:
                var_attr["coordinates"] = "idx lat lon"

        return var_attr

    def add_ts_results(self, idx, times, results, attr=None):
        """
        Add observations over time to previously added locations results.

        Parameters
        ----------
        idx : int
            Location index, as returned when adding metrics results.
        times : pd.DatetimeIndex or np.array
            Datetime index as in the validation results from rolling metrics
        results : dict
            Variable names as dict keys and data arrays as values.
            Data arrays must have same size as times.
        attr : dict, optional (default: None)
            Variable names as keys and attributes as dicts for each variable.
            Only used when the variable is created, not if it is already in the
            dataset.
        """
        sel = slice(self["_row_size"][:][:idx].sum(), None)
        self["time"][sel] = self._date2num(times)
        self["_row_size"][idx] = len(times)  # update the time series length

        for name, data in results.items():
            if name not in self.variables:
                var_attr = self._sel_attr(attr, name, False)
                self.write_var(
                    name, data=data, dim=("obs",), attr=var_attr
                )
            else:
                self[name][sel] = data

    def add_metrics_results(self, lons, lats, results, attr=None):
        """
        Add observations over time to a locations results.

        Parameters
        ----------
        lons : np.array
            Array of location longitudes, shape must match shape of arrays
            in data
        lats : np.array
            Array of location latitudes, shape must match shape of arrays
            in data
        results : dict
            Variable names as dict keys and data arrays as values. As returned
            by the metric calculators, except the RollingMetrics. Shape of data
            arrays must match lons/lats.
        attr : dict, optional (default: None)
            Variable names as keys and attributes as dicts for each variable.
            Only used when the variable is created, not if it is already in the
            dataset.

        Returns
        -------
        idx : indices
            Indices of the new locations, can be used to add time results
        """
        n = len(lons)
        if not np.all(
            np.array(
                [len(lons), len(lats)] + [len(d) for d in results.values()]
            )
            == n
        ):
            raise ValueError(
                "Lon, Lat and data variable must have the same shape"
            )

        idx = self["idx"][:][-1] + 1 if len(self["idx"]) != 0 else 0
        sel = slice(idx, None)

        self["idx"][sel] = np.arange(idx, idx + n)  # + 1
        self["_row_size"][sel] = 0
        self["lon"][sel], self["lat"][sel] = lons, lats

        for name, data in results.items():
            if isinstance(name, tuple):
                name = "__".join(name)
            if name not in self.variables:
                var_attr = self._sel_attr(attr, name, False)
                self.write_var(name, data=data, dim=("loc",), attr=var_attr)
            else:
                self[name][sel] = data

        return self["idx"][sel].data

    def add_result(self, lon, lat, data, ts_vars=None, times=None, attr=None):
        """
        Add all results (time series and location metrics) for a single point.

        Parameters
        ----------
        lon : float
            Longitude of the point
        lat : float
            Latitude of the point
        data : dict
            Dict of metric names and values. For normal (not rolling) metrics
            this is an array of size 1, otherwise if the same size as time.
        times : np.array, optional (default: None)
            Time values, length must mach all time series (rolling) metrics
            in data.
        attr : dict, optional (default: None)
        """

        if attr is None:
            attr = {}
        if ts_vars is None:
            ts_vars = []

        metric_results = {k: v for k, v in data.items() if k not in ts_vars}
        metric_attrs = {
            k: v for k, v in attr.items() if k in metric_results.keys()
        }

        ts_results = {k: v for k, v in data.items() if k in ts_vars}

        if times is None and ts_results:
            raise ValueError(
                "Got time series variables, "
                "but no times passed: {}".format(ts_results.keys())
            )

        idx = self.add_metrics_results(
            lons=np.array([lon]),
            lats=np.array([lat]),
            results=metric_results,
            attr=metric_attrs,
        )

        if ts_results:
            assert len(idx) == 1
            ts_attrs = {
                k: v for k, v in attr.items() if k in ts_results.keys()
            }
            self.add_ts_results(
                idx[0], times=times, results=ts_results, attr=ts_attrs
            )

    def read_ts(self, idx: int) -> pd.DataFrame:
        """Read time series data for a single point"""
        var_data = {}
        sel = None
        for var in self.variables:
            if var in ["time", "idx", "_row_size"]:
                continue
            if "obs" in self[var].dimensions:
                if sel is None:
                    sel_start = self["_row_size"][:][:idx].sum()
                    sel = slice(
                        sel_start, sel_start + self["_row_size"][:][idx]
                    )
                var_data[var] = self[var][sel]

        return pd.DataFrame(
            index=self._num2date(self["time"][sel]), data=var_data
        )

    def read_loc(self, idx: Union[int, np.array, None] = None) -> pd.DataFrame:
        """Read loc data for one/multiple/all point(s)"""
        if idx is None:
            data = pd.DataFrame(index=self["idx"][:])
            idx = slice(None, None)
        else:
            if isinstance(idx, int):
                idx = [idx]
            data = pd.DataFrame(index=idx)

        for var in self.variables:
            if var in ["time", "idx", "_row_size"]:
                continue
            if "obs" not in self[var].dimensions:
                data[var] = self[var][:][idx]

        return data


def netcdf_results_manager(
    results, save_path, filename: dict = None,
    ts_vars: list = None, zlib=True, attr=None
):
    """
    Write validation results to netcdf file.

    Parameters
    ----------
    results : dict
        Validation results as returned by the metrics calculator.
        Keys are tuples that define the dataset names that were used.
        Values contains 'lon' and 'lat' keys for defining the points, and
        optionally 'time' which sets the time stamps for each location
        (if there are metrics over time in the results - e.g due to
        RollingMetrics)
    save_path : str
        Directory where the netcdf file(s) are are created.
    filename: dict, optional (default: None)
        Filename(s) (value), for each dataset combination in results (key).
        By default (if None is passed) the keys in results are used to
        generate a file name.
    ts_vars : list, optional (default: None)
        List of variables in results that are treated as time series
    zlib : bool, optional (default: True)
        Activate compression
    attr : dict, optional (default: None)
        Variable attributes, variable names as keys, attributes as another
        dict in values.
    """

    if len(results) == 0:
        warnings.warn(f"Empty results, {save_path} will not be created.")
    for ds_names, res in results.items():
        if filename is None:
            fname = build_filename(save_path, ds_names)
        else:
            fname = os.path.join(save_path, filename[ds_names])

        with PointDataResults(fname, zlib=zlib) as writer:
            lons = res.pop("lon")
            lats = res.pop("lat")
            if ts_vars is not None:
                for i, (lon, lat) in enumerate(zip(lons, lats)):
                    data = {}
                    for k, v in res.items():
                        if k.lower() == "time":
                            time = pd.DatetimeIndex(res["time"][i])
                        elif k in ts_vars:
                            data[k] = v[i]
                        else:
                            if not isinstance(v[i], np.ndarray):
                                data[k] = np.array([v[i]])
                            else:
                                data[k] = v[i]

                    writer.add_result(
                        lon,
                        lat,
                        data=data,
                        ts_vars=ts_vars,
                        times=time,
                        attr=attr,
                    )

            else:
                writer.add_metrics_results(lons, lats, results=res, attr=attr)
