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
Module containing adapters that can be used together with the validation
framework.
"""

import operator
from pytesmo.time_series.anomaly import calc_anomaly
from pytesmo.time_series.anomaly import calc_climatology
from pytesmo.utils import deprecated
from pandas import DataFrame
import warnings

_op_lookup = {
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
    ">=": operator.ge,
    ">": operator.gt,
}


class BasicAdapter:
    """
    Adapter to modify the return value of reading functions from base class.
    - Pick data frame from objects that have a `data_property_name`,
      i.e. ascat time series objects.
    - Removes unnecessary timezone information in pandas data frames which
      pytesmo can not use.
    - adds a method with the name given in `read_name` that calls the same
      method from cls but modifies the returned data frame.
    """

    def __init__(self, cls, data_property_name="data", read_name=None):
        """
        Parameters
        ----------
        cls: object
            The original reader to adapt.
        data_property_name: str, optional (default: "data")
            Attribute name under which the pandas DataFrame containing the time
            series is found in the object returned by the read function of the
            original reader. Ignored if no attribute of this name is found.
            Then it is required that the DataFrame is already the return value
            of the read function.
        read_name: str, optional (default: None)
            To enable the adapter for a method other than `read` or `read_ts`
            give the function name here (a function of that name must exist in
            cls). A method of the same name will be added to the adapted
            Reader, which takes the same arguments as the base method.
            The output of this method will be changed by the adapter.
            If None is passed, only data from `read` and `read_ts` of cls
            will be adapted.
        """

        self.cls = cls
        self.data_property_name = data_property_name
        self.read_name = read_name

        if read_name:
            setattr(self, read_name, self._adapt_custom)

    def __get_dataframe(self, data):
        if (
            (not isinstance(data, DataFrame))
            and (hasattr(data, self.data_property_name))
            and (isinstance(getattr(data, self.data_property_name), DataFrame))
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

    def _adapt(self, df: DataFrame) -> DataFrame:
        # drop time zone info and extract df from ASCAT TS object
        return self.__drop_tz_info(self.__get_dataframe(df))

    def _adapt_custom(self, *args, **kwargs):
        # modifies data from whatever function was set as `read_name`.
        data = getattr(self.cls, self.read_name)(*args, **kwargs)
        return self._adapt(data)

    def read_ts(self, *args, **kwargs):
        data = getattr(self.cls, "read_ts")(*args, **kwargs)
        return self._adapt(data)

    def read(self, *args, **kwargs):
        data = getattr(self.cls, "read")(*args, **kwargs)
        return self._adapt(data)

    @property
    def grid(self):
        """
        Returns grid of wrapped class if it exists, otherwise None.
        """
        if hasattr(self.cls, "grid"):
            return self.cls.grid


@deprecated(
    "`MaskingAdapter` is deprecated, use `SelfMaskingAdapter` "
    "or `AdvancedMaskingAdapter` instead."
)
class MaskingAdapter(BasicAdapter):
    """
    Transform the given class to return a boolean dataset given the operator
    and threshold. This class calls the read_ts and read methods
    of the given instance and applies boolean masking to the returned data
    using the given operator and threshold.

    Parameters
    ----------
    cls: object
        Reader object, has to have a `read_ts` or `read` method or a method
        name must be specified in the `read_name` kwarg. The same method will
        be available for the adapted version of the reader.
    op: str or Callable
        Either a string to look up a function from
        :const:`pytesmo/validation_framework/adapters.py._op_lookup`
        or a function that takes `data` and `threshold` as arguments.
    threshold: Any
        Value to use as the threshold combined with the operator to mask
        elements in `column_name`
    column_name: str, optional (default: None)
        Name of the column to apply `op` to. If None is passed,
        nothing happens.
    data_property_name: str, optional (default: "data")
        Attribute name under which the pandas DataFrame containing the time
        series is found in the object returned by the read function of the
        original reader. Ignored if no attribute of this name is found.
        Then it is required that the DataFrame is already the return value
        of the read function.
    read_name: str, optional (default: None)
        To enable the adapter for a method other than `read` or `read_ts`
        give the function name here (a function of that name must exist in
        cls). A method of the same name will be added to the adapted
        Reader, which takes the same arguments as the base method.
        The output of this method will be changed by the adapter.
        If None is passed, only data from `read` and `read_ts` of cls
        will be adapted.
    """

    def __init__(self, cls, op, threshold, column_name=None, **kwargs):

        super().__init__(cls, **kwargs)

        if callable(op):
            self.operator = op
        elif op in _op_lookup:
            self.operator = _op_lookup[op]
        else:
            raise ValueError('"{}" is not a valid operator'.format(op))

        self.threshold = threshold

        self.column_name = column_name

    def _adapt(self, data):
        data = super()._adapt(data)
        if self.column_name is not None:
            data = data.loc[:, [self.column_name]]
        return self.operator(data, self.threshold)


class SelfMaskingAdapter(BasicAdapter):
    """
    Transform the given (reader) class to return a dataset that is masked based
    on the given column, operator, and threshold. This class calls the read_ts
    or read method of the given reader instance, applies the operator/threshold
    to the specified column, and masks the whole dataframe with the result.

    Parameters
    ----------
    cls: object
        Reader object, has to have a `read_ts` or `read` method or a method
        name must be specified in the `read_name` kwarg. The same method will
        be available for the adapted version of the reader.
    op: str or Callable
        Either a string to look up a function from
        :const:`pytesmo/validation_framework/adapters.py._op_lookup`
        or a function that takes `data` and `threshold` as arguments.
    threshold: Any
        Value to use as the threshold combined with the operator to mask
        elements in `column_name`
    column_name: str
        Name of the column to apply `op` to
    data_property_name: str, optional (default: "data")
        Attribute name under which the pandas DataFrame containing the time
        series is found in the object returned by the read function of the
        original reader. Ignored if no attribute of this name is found.
        Then it is required that the DataFrame is already the return value
        of the read function.
    read_name: str, optional (default: None)
        To enable the adapter for a method other than `read` or `read_ts`
        give the function name here (a function of that name must exist in
        cls). A method of the same name will be added to the adapted
        Reader, which takes the same arguments as the base method.
        The output of this method will be changed by the adapter.
        If None is passed, only data from `read` and `read_ts` of cls
        will be adapted.
    """

    def __init__(self, cls, op, threshold, column_name, **kwargs):

        super().__init__(cls, **kwargs)

        if callable(op):
            self.operator = op
        elif op in _op_lookup:
            self.operator = _op_lookup[op]
        else:
            raise ValueError(f"'{op}' is not a valid operator")

        self.threshold = threshold
        self.column_name = column_name

    def _adapt(self, data):
        data = super()._adapt(data)
        mask = self.operator(data[self.column_name], self.threshold)
        return data[mask]


class AdvancedMaskingAdapter(BasicAdapter):
    """
    Transform the given (reader) class to return a dataset that is masked based
    on the given list of filters. A filter is a 3-tuple of column_name,
    operator, and threshold.
    This class calls the reading method of the given reader instance,
    applies all filters separately, ANDs all filters together, and masks the
    whole dataframe with the result.

    Parameters
    ----------
    cls: object
        Reader object, has to have a `read_ts` or `read` method or a method
        name must be specified in the `read_name` kwarg. The same method will
        be available for the adapted version of the reader.
    filter_list: list of 3-tuples: column_name, operator, and threshold.
        'column_name': string
            name of the column to apply the operator to
        'operator': Callable or str;
            string needs to be one of '<', '<=', '==', '>=', '>', '!=' or a
            function that takes data and threshold as arguments.
        'threshold':
            value to use as the threshold combined with the operator;
    data_property_name: str, optional (default: "data")
        Attribute name under which the pandas DataFrame containing the time
        series is found in the object returned by the read function of the
        original reader. Ignored if no attribute of this name is found.
        Then it is required that the DataFrame is already the return value
        of the read function.
    read_name: str, optional (default: None)
        To enable the adapter for a method other than `read` or `read_ts`
        give the function name here (a function of that name must exist in
        cls). A method of the same name will be added to the adapted
        Reader, which takes the same arguments as the base method.
        The output of this method will be changed by the adapter.
        If None is passed, only data from `read` and `read_ts` of cls
        will be adapted.
    """

    def __init__(self, cls, filter_list, **kwargs):

        super().__init__(cls, **kwargs)

        self.filter_list = filter_list

    def _adapt(self, data):
        data = super()._adapt(data)
        mask = None
        for column_name, op, threshold in self.filter_list:
            if callable(op):
                operator = op
            elif op in _op_lookup:
                operator = _op_lookup[op]
            else:
                raise ValueError('"{}" is not a valid operator'.format(op))

            new_mask = operator(data[column_name], threshold)

            if mask is not None:
                mask = mask & new_mask
            else:
                mask = new_mask

        return data[mask]


class AnomalyAdapter(BasicAdapter):
    """
    Takes the pandas DataFrame that reader returns and calculates the
    anomaly of the time series based on a moving average.

    Parameters
    ----------
    cls: object
        Reader object, has to have a `read_ts` or `read` method or a method
        name must be specified in the `read_name` kwarg. The same method will
        be available for the adapted version of the reader.
    window_size : float, optional (default: 35)
        The window-size [days] of the moving-average window to calculate the
        anomaly reference.
    columns: list, optional
        columns in the dataset for which to calculate anomalies.
    data_property_name: str, optional (default: "data")
        Attribute name under which the pandas DataFrame containing the time
        series is found in the object returned by the read function of the
        original reader. Ignored if no attribute of this name is found.
        Then it is required that the DataFrame is already the return value
        of the read function.
    read_name: str, optional (default: None)
        To enable the adapter for a method other than `read` or `read_ts`
        give the function name here (a function of that name must exist in
        cls). A method of the same name will be added to the adapted
        Reader, which takes the same arguments as the base method.
        The output of this method will be changed by the adapter.
        If None is passed, only data from `read` and `read_ts` of cls
        will be adapted.
    """

    def __init__(self, cls, window_size=35, columns=None, **kwargs):

        super().__init__(cls, **kwargs)

        self.window_size = window_size
        self.columns = columns

    def _adapt(self, data):
        data = super()._adapt(data)
        if self.columns is None:
            ite = data
        else:
            ite = self.columns
        for column in ite:
            data[column] = calc_anomaly(
                data[column], window_size=self.window_size
            )
        return data


class AnomalyClimAdapter(BasicAdapter):
    """
    Takes the pandas DataFrame that reader returns and calculates the
    anomaly of the time series based on the (long-term) average of the series.

    Parameters
    ----------
    cls: object
        Reader object, has to have a `read_ts` or `read` method or a method
        name must be specified in the `read_name` kwarg. The same method will
        be available for the adapted version of the reader.
    columns: list, optional (default: None)
        Columns in the dataset for which to calculate anomalies. If None is
        passed, the anomaly is calculated for all columns.
    data_property_name: str, optional (default: "data")
        Attribute name under which the pandas DataFrame containing the time
        series is found in the object returned by the read function of the
        original reader. Ignored if no attribute of this name is found.
        Then it is required that the DataFrame is already the return value
        of the read function.
    read_name: str, optional (default: None)
        To enable the adapter for a method other than `read` or `read_ts`
        give the function name here (a function of that name must exist in
        cls). A method of the same name will be added to the adapted
        Reader, which takes the same arguments as the base method.
        The output of this method will be changed by the adapter.
        If None is passed, only data from `read` and `read_ts` of cls
        will be adapted.
    kwargs:
        Any remaining keyword arguments will be given to
        :func:`pytesmo.time_series.anomaly.calc_climatology`
    """

    def __init__(self, cls, columns=None, **kwargs):

        cls_kwargs = dict()
        if "data_property_name" in kwargs:
            cls_kwargs["data_property_name"] = kwargs.pop("data_property_name")
        if "read_name" in kwargs:
            cls_kwargs["read_name"] = kwargs.pop("read_name")

        super().__init__(cls, **cls_kwargs)

        self.kwargs = kwargs
        self.columns = columns

    def _adapt(self, data):
        data = super()._adapt(data)
        if self.columns is None:
            ite = data
        else:
            ite = self.columns
        for column in ite:
            clim = calc_climatology(data[column], **self.kwargs)
            data[column] = calc_anomaly(data[column], climatology=clim)

        return data


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
        cls : object
            Reader object, has to have a `read_ts` or `read` method or a
            method name must be specified in the `read_name` kwarg.
            The same method will be available for the adapted version of the
            reader.
        func: Callable
            Will be applied to dataframe columns using
            pd.DataFrame.apply(..., axis=1)
            additional kwargs for this must be given in func_kwargs,
            e.g. :func:`pd.DataFrame.mean`
        func_kwargs : dict, optional (default: None)
            kwargs that are passed to method or None to use the default ones.
        columns: list, optional (default: None)
            Columns in the dataset that are combined. If None are selected
            all columns are used.
        new_name: str, optional (default: merged)
            Name that the merged column will have in the returned data frame.
        data_property_name: str, optional (default: "data")
            Attribute name under which the pandas DataFrame containing the
            time series is found in the object returned by the read function
            of the original reader.
            Ignored if no attribute of this name is found.
            Then it is required that the DataFrame is already the return value
            of the read function.
        read_name: str, optional (default: None)
            To enable the adapter for a method other than `read` or `read_ts`
            give the function name here (a function of that name must exist in
            cls). A method of the same name will be added to the adapted
            Reader, which takes the same arguments as the base method.
            The output of this method will be changed by the adapter.
            If None is passed, only data from `read` and `read_ts` of cls
            will be adapted.
        """

        super().__init__(cls, **kwargs)
        self.func = func
        self.func_kwargs = func_kwargs if func_kwargs is not None else {}
        self.func_kwargs["axis"] = 1
        self.columns = columns
        self.new_name = new_name

    def _adapt(self, data: DataFrame) -> DataFrame:
        data = super()._adapt(data)
        columns = data.columns if self.columns is None else self.columns
        new_col = data[columns].apply(self.func, **self.func_kwargs)
        data[self.new_name] = new_col
        return data
