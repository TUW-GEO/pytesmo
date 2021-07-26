# Copyright (c) 2016,Vienna University of Technology,
# Department of Geodesy and Geoinformation
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#   * Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#   * Neither the name of the Vienna University of Technology, Department of
#     Geodesy and Geoinformation nor the names of its contributors may be used
#     to endorse or promote products derived from this software without
#     specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL VIENNA UNIVERSITY OF TECHNOLOGY, DEPARTMENT
# OF GEODESY AND GEOINFORMATION BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Module containing adapters that can be used together with the validation
framework.
"""

import operator
from pytesmo.time_series.anomaly import calc_anomaly
from pytesmo.time_series.anomaly import calc_climatology
from pandas import DataFrame
import warnings
from pytesmo.utils import deprecated


class BasicAdapter:
    """
    Adapter to harmonise certain parts of base reader classes.

     - Pick data frame from objects that have a `data_property_name`,
     i.e. ascat time series.
     - Removes unnecessary timezone information in pandas data frames which
     pytesmo can not use.
     - Maps the `read` function of the adapted reader to a different function
     of the base class as specified in `read_name`.
    """

    def __init__(self, cls, data_property_name="data", read_name="read"):
        """
        Parameters
        ----------
        cls: object
            The original reader to adapt.
        data_property_name: str
            Attribute name under which the pandas DataFrame containing the time
            series is found in the object returned by the read function of the
            original reader.
        read_name: str, optional (default: 'read')
            Function name(s) in the original reader that is mapped to the
            adapters 'read' function. By default 'read' is used.
        """

        self.cls = cls
        self.data_property_name = data_property_name
        self.read_name = read_name

    def __get_dataframe(self, data):
        if (
            (not isinstance(data, DataFrame))
            and (hasattr(data, self.data_property_name))
            and (
                isinstance(getattr(data, self.data_property_name), DataFrame)
            )
        ):
            data = getattr(data, self.data_property_name)
        return data

    def __drop_tz_info(self, data):
        if hasattr(data.index, "tz") and (data.index.tz is not None):
            warnings.warn(
                f"Dropping timezone information ({data.index.tz})"
                f" for data from reader {self.cls.__class__.__name__}"
            )
            data.index = data.index.tz_convert(None)
        return data

    def __read(self, *args, **kwargs):
        # calls whatever function was set as `read_name`, default: `read()`
        data = getattr(self.cls, self.read_name)(*args, **kwargs)
        data = self.__drop_tz_info(self.__get_dataframe(data))
        return data

    @deprecated(
        "`read_ts` is deprecated, use `read` instead."
        "To map to a method other than `read` specify it in "
        "`read_name`."
    )
    def read_ts(self, *args, **kwargs):
        data = self.cls.read_ts(*args, **kwargs)
        data = self.__drop_tz_info(self.__get_dataframe(data))
        return data

    def read(self, *args, **kwargs):
        return self.__read(*args, **kwargs)

    @property
    def grid(self):
        """
        Returns grid of wrapped class if it exists, otherwise None.
        """
        if hasattr(self.cls, "grid"):
            return self.cls.grid


class MaskingAdapter(BasicAdapter):
    """
    Transform the given class to return a boolean dataset given the operator
    and threshold. This class calls the read_ts and read methods
    of the given instance and applies boolean masking to the returned data
    using the given operator and threshold.

    Parameters
    ----------
    cls: object
        has to have a read_ts or read method
    op: str or Callable
        one of '<', '<=', '==', '>=', '>', '!=' or a function that takes
        data and threshold as arguments.
    threshold:
        value to use as the threshold combined with the operator
    column_name: str, optional
        name of the column to cut the read masking dataset to
    kwargs:
        Additional kwargs are passed to BasicAdapter.
    """

    def __init__(self, cls, op, threshold, column_name=None, **kwargs):
        super(MaskingAdapter, self).__init__(cls, **kwargs)

        self.op_lookup = {
            "<": operator.lt,
            "<=": operator.le,
            "==": operator.eq,
            "!=": operator.ne,
            ">=": operator.ge,
            ">": operator.gt,
        }

        if callable(op):
            self.operator = op
        elif op in self.op_lookup:
            self.operator = self.op_lookup[op]
        else:
            raise ValueError('"{}" is not a valid operator'.format(op))

        self.threshold = threshold

        self.column_name = column_name

    def __mask(self, data):
        if self.column_name is not None:
            data = data.loc[:, [self.column_name]]
        return self.operator(data, self.threshold)

    @deprecated(
        "`read_ts` is deprecated, use `read` instead."
        "To map to a method other than `read` specify it in "
        "`read_name`."
    )
    def read_ts(self, *args, **kwargs):
        data = super(MaskingAdapter, self).read_ts(*args, **kwargs)
        return self.__mask(data)

    def read(self, *args, **kwargs):
        data = super(MaskingAdapter, self).read(*args, **kwargs)
        return self.__mask(data)


class SelfMaskingAdapter(BasicAdapter):
    """
    Transform the given (reader) class to return a dataset that is masked based
    on the given column, operator, and threshold. This class calls the read_ts
    or read method of the given reader instance, applies the operator/threshold
    to the specified column, and masks the whole dataframe with the result.

    Parameters
    ----------
    cls: object
        has to have a read_ts or read method
    op: Callable or str
        one of '<', '<=', '==', '>=', '>', '!=' or a function that takes
        data and threshold as arguments.
    threshold:
        value to use as the threshold combined with the operator
    column_name: str
        name of the column to apply the threshold to
    kwargs:
        Additional kwargs are passed to BasicAdapter.
    """

    def __init__(self, cls, op, threshold, column_name, **kwargs):
        super(SelfMaskingAdapter, self).__init__(cls, **kwargs)

        self.op_lookup = {
            "<": operator.lt,
            "<=": operator.le,
            "==": operator.eq,
            "!=": operator.ne,
            ">=": operator.ge,
            ">": operator.gt,
        }

        if callable(op):
            self.operator = op
        elif op in self.op_lookup:
            self.operator = self.op_lookup[op]
        else:
            raise ValueError('"{}" is not a valid operator'.format(op))

        self.threshold = threshold
        self.column_name = column_name

    def __mask(self, data):
        mask = self.operator(data[self.column_name], self.threshold)
        return data[mask]

    @deprecated(
        "`read_ts` is deprecated, use `read` instead."
        "To map to a method other than `read` specify it in "
        "`read_name`."
    )
    def read_ts(self, *args, **kwargs):
        data = super(SelfMaskingAdapter, self).read_ts(*args, **kwargs)
        return self.__mask(data)

    def read(self, *args, **kwargs):
        data = super(SelfMaskingAdapter, self).read(*args, **kwargs)
        return self.__mask(data)


class AdvancedMaskingAdapter(BasicAdapter):
    """
    Transform the given (reader) class to return a dataset that is masked based
    on the given list of filters. A filter is a 3-tuple of column_name,
    operator, and threshold.
    This class calls the read_ts or read method of the given reader instance,
    applies all filters separately, ANDs all filters together, and masks the
    whole dataframe with the result.

    Parameters
    ----------
    cls: object
        has to have a read_ts or read method, if not specify a different method
        name that is then mapped to `self.read` using the `read_name` kwarg.
    filter_list: list of 3-tuples: column_name, operator, and threshold.
        'column_name': string
            name of the column to apply the operator to
        'operator': Callable or str;
            string needs to be one of '<', '<=', '==', '>=', '>', '!=' or a
            function that takes data and threshold as arguments.
        'threshold':
            value to use as the threshold combined with the operator;
    kwargs:
        Additional kwargs are passed to BasicAdapter.
    """

    def __init__(self, cls, filter_list, **kwargs):
        super(AdvancedMaskingAdapter, self).__init__(cls, **kwargs)

        self.op_lookup = {
            "<": operator.lt,
            "<=": operator.le,
            "==": operator.eq,
            "!=": operator.ne,
            ">=": operator.ge,
            ">": operator.gt,
        }

        self.filter_list = filter_list

    def __mask(self, data):
        mask = None
        for column_name, op, threshold in self.filter_list:
            if callable(op):
                operator = op
            elif op in self.op_lookup:
                operator = self.op_lookup[op]
            else:
                raise ValueError('"{}" is not a valid operator'.format(op))

            new_mask = operator(data[column_name], threshold)

            if mask is not None:
                mask = mask & new_mask
            else:
                mask = new_mask

        return data[mask]

    @deprecated(
        "`read_ts` is deprecated, use `read` instead."
        "To map to a method other than `read` specify it in "
        "`read_name`."
    )
    def read_ts(self, *args, **kwargs):
        data = super(AdvancedMaskingAdapter, self).read_ts(*args, **kwargs)
        return self.__mask(data)

    def read(self, *args, **kwargs):
        data = super(AdvancedMaskingAdapter, self).read(*args, **kwargs)
        return self.__mask(data)


class AnomalyAdapter(BasicAdapter):
    """
    Takes the pandas DataFrame that the read_ts or read method of the instance
    returns and calculates the anomaly of the time series based on a moving
    average.

    Parameters
    ----------
    cls : class instance
        Must have a read_ts or read method returning a pandas.DataFrame or the
        name of the method to map to `read` in the adapted version of cls must
        must be set.
    window_size : float, optional
        The window-size [days] of the moving-average window to calculate the
        anomaly reference (only used if climatology is not provided)
        Default: 35 (days)
    columns: list, optional
        columns in the dataset for which to calculate anomalies.
    kwargs:
        Additional kwargs are passed to BasicAdapter.
    """

    def __init__(self, cls, window_size=35, columns=None, **kwargs):
        super(AnomalyAdapter, self).__init__(cls, **kwargs)
        self.window_size = window_size
        self.columns = columns

    def calc_anom(self, data):
        if self.columns is None:
            ite = data
        else:
            ite = self.columns
        for column in ite:
            data[column] = calc_anomaly(
                data[column], window_size=self.window_size
            )
        return data

    @deprecated(
        "`read_ts` is deprecated, use `read` instead."
        "To map to a method other than `read` specify it in "
        "`read_name`."
    )
    def read_ts(self, *args, **kwargs):
        data = super(AnomalyAdapter, self).read_ts(*args, **kwargs)
        return self.calc_anom(data)

    def read(self, *args, **kwargs):
        data = super(AnomalyAdapter, self).read(*args, **kwargs)
        return self.calc_anom(data)


class AnomalyClimAdapter(BasicAdapter):
    """
    Takes the pandas DataFrame that the read_ts or read method of the instance
    returns and calculates the anomaly of the time series based on a moving
    average.

    Parameters
    ----------
    cls: class instance
        Must have a read_ts or read method returning a pandas.DataFrame, or a
        method name to map to read() must be specified in cls_kwargs
    cls_kwargs: dict, optional (default: None)
        Kwargs that are passed to create BasicAdapter.
    columns: list, optional
        columns in the dataset for which to calculate anomalies.
    kwargs:
        Any additional arguments will be given to the calc_climatology
        function.
        If 'data_property_name' or 'read_name' are in kwargs, they will be
        used to initialise the BasicAdapter.
    """

    def __init__(self, cls, columns=None, **kwargs):

        cls_kwargs = dict()
        if "data_property_name" in kwargs:
            cls_kwargs["data_property_name"] = kwargs.pop(
                "data_property_name"
            )
        if "read_name" in kwargs:
            cls_kwargs["read_name"] = kwargs.pop("read_name")

        super(AnomalyClimAdapter, self).__init__(cls, **cls_kwargs)

        self.kwargs = kwargs
        self.columns = columns

    def calc_anom(self, data):
        if self.columns is None:
            ite = data
        else:
            ite = self.columns
        for column in ite:
            clim = calc_climatology(data[column], **self.kwargs)
            data[column] = calc_anomaly(data[column], climatology=clim)
        return data

    @deprecated(
        "`read_ts` is deprecated, use `read` instead."
        "To map to a method other than `read` specify it in "
        "`read_name`."
    )
    def read_ts(self, *args, **kwargs):
        data = super(AnomalyClimAdapter, self).read_ts(*args, **kwargs)
        return self.calc_anom(data)

    def read(self, *args, **kwargs):
        data = super(AnomalyClimAdapter, self).read(*args, **kwargs)
        return self.calc_anom(data)


class ColumnCombineAdapter(BasicAdapter):
    """
    Takes the pandas DataFrame that the read_ts or read method of the instance
    returns and applies a function to merge multiple columns into one.
    E.g. when there are 2 Soil Moisture parameters in a dataset that should be
    averaged on reading. Will add one additional column to the input data
    frame.
    """

    def __init__(
        self,
        cls,
        func,
        func_kwargs=None,
        columns=None,
        new_name="merged",
        **kwargs,
    ):
        """
        Parameters
        ----------
        cls : class instance
            Must have a read_ts or read method returning a pandas.DataFrame
        func: Callable
            Will be applied to dataframe columns using
            pd.DataFrame.apply(..., axis=1)
            additional kwargs for this must be given in func_kwargs,
            e.g. pd.DataFrame.mean
        func_kwargs : dict, optional (default: None)
            kwargs that are passed to method
        columns: list, optional (default: None)
            Columns in the dataset that are combined. If None are selected
            all columns are used.
        new_name: str, optional (default: merged)
            Name that the merged column will have in the returned data frame.
        kwargs:
            Additional kwargs are passed to BasicAdapter.
        """

        super(ColumnCombineAdapter, self).__init__(cls, **kwargs)
        self.func = func
        self.func_kwargs = func_kwargs if func_kwargs is not None else {}
        self.func_kwargs["axis"] = 1
        self.columns = columns
        self.new_name = new_name

    def apply(self, data: DataFrame) -> DataFrame:
        columns = data.columns if self.columns is None else self.columns
        new_col = data[columns].apply(self.func, **self.func_kwargs)
        data[self.new_name] = new_col
        return data

    def read_ts(self, *args, **kwargs) -> DataFrame:
        data = super(ColumnCombineAdapter, self).read_ts(*args, **kwargs)
        return self.apply(data)

    def read(self, *args, **kwargs) -> DataFrame:
        return self.read_ts(*args, **kwargs)
