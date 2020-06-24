# Copyright (c) 2016,Vienna University of Technology,
# Department of Geodesy and Geoinformation
# All rights reserved.
import warnings

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#   * Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#   * Neither the name of the Vienna University of Technology, Department of
#     Geodesy and Geoinformation nor the names of its contributors may be used
#     to endorse or promote products derived from this software without specific
#     prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL VIENNA UNIVERSITY OF TECHNOLOGY, DEPARTMENT OF
# GEODESY AND GEOINFORMATION BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

'''
Module containing adapters that can be used together with the validation
framework.
'''

import operator
from pytesmo.time_series.anomaly import calc_anomaly
from pytesmo.time_series.anomaly import calc_climatology
from pandas import DataFrame


class BasicAdapter(object):
    """
    Base class for other adapters that works around data readers that don't
    return a DataFrame (e.g. ASCAT). Also removes unnecessary timezone information in data.
    """
    def __init__(self, cls, data_property_name='data'):
        self.cls = cls
        self.data_property_name = data_property_name

    def __get_dataframe(self, data):
        if ((not isinstance(data, DataFrame)) and (hasattr(data, self.data_property_name)) and
            (isinstance(getattr(data, self.data_property_name), DataFrame))):
            data = getattr(data, self.data_property_name)
        return data

    def __drop_tz_info(self, data):
        if (hasattr(data.index, 'tz') and (data.index.tz is not None)):
            warnings.warn('Dropping timezone information ({}) for data from reader {}'.format(data.index.tz, self.cls.__class__.__name__))
            data.index = data.index.tz_convert(None)
        return data

    def read_ts(self, *args, **kwargs):
        data = self.cls.read_ts(*args, **kwargs)
        data = self.__drop_tz_info(self.__get_dataframe(data))
        return data

    def read(self, *args, **kwargs):
        data = self.cls.read(*args, **kwargs)
        data = self.__drop_tz_info(self.__get_dataframe(data))
        return data

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
    operator: string
        one of '<', '<=', '==', '>=', '>', '!='
    threshold:
        value to use as the threshold combined with the operator
    column_name: string, optional
        name of the column to cut the read masking dataset to
    """

    def __init__(self, cls, op, threshold, column_name = None):
        super(MaskingAdapter, self).__init__(cls)

        self.op_lookup = {'<': operator.lt,
                          '<=': operator.le,
                          '==': operator.eq,
                          '!=': operator.ne,
                          '>=': operator.ge,
                          '>': operator.gt}
        self.operator = self.op_lookup[op]
        self.threshold = threshold

        self.column_name = column_name

    def __mask(self, data):
        if self.column_name is not None:
            data = data.loc[:, [self.column_name]]
        return self.operator(data, self.threshold)

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
    operator: string
        one of '<', '<=', '==', '>=', '>', '!='
    threshold:
        value to use as the threshold combined with the operator
    column_name: string
        name of the column to apply the threshold to
    """

    def __init__(self, cls, op, threshold, column_name):
        super(SelfMaskingAdapter, self).__init__(cls)

        self.op_lookup = {'<': operator.lt,
                          '<=': operator.le,
                          '==': operator.eq,
                          '!=': operator.ne,
                          '>=': operator.ge,
                          '>': operator.gt}

        self.operator = self.op_lookup[op]
        self.threshold = threshold
        self.column_name = column_name

    def __mask(self, data):
        mask = self.operator(data[self.column_name], self.threshold)
        return data[mask]

    def read_ts(self, *args, **kwargs):
        data = super(SelfMaskingAdapter, self).read_ts(*args, **kwargs)
        return self.__mask(data)

    def read(self, *args, **kwargs):
        data = super(SelfMaskingAdapter, self).read(*args, **kwargs)
        return self.__mask(data)


class AdvancedMaskingAdapter(BasicAdapter):
    """
    Transform the given (reader) class to return a dataset that is masked based
    on the given list of filters. A filter is a 3-tuple of column_name, operator, and threshold.
    This class calls the read_ts or read method of the given reader instance, applies all filters
    separately, ANDs all filters together, and masks the whole dataframe with the result.

    Parameters
    ----------
    cls: object
        has to have a read_ts or read method
    filter_list: list of 3-tuples: column_name, operator, and threshold.
        'column_name': string
            name of the column to apply the operator to
        'operator': callable or string;
            callable needs to accept two parameters (e.g. my_mask(a,b);
            the threshold will be passed to the second parameter
            string needs to be one of '<', '<=', '==', '>=', '>', '!='
        'threshold':
            value to use as the threshold combined with the operator;
    """

    def __init__(self, cls, filter_list):
        super(AdvancedMaskingAdapter, self).__init__(cls)

        self.op_lookup = {'<': operator.lt,
                          '<=': operator.le,
                          '==': operator.eq,
                          '!=': operator.ne,
                          '>=': operator.ge,
                          '>': operator.gt}

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
        Must have a read_ts or read method returning a pandas.DataFrame
    window_size : float, optional
        The window-size [days] of the moving-average window to calculate the
        anomaly reference (only used if climatology is not provided)
        Default: 35 (days)
    columns: list, optional
        columns in the dataset for which to calculate anomalies.
    """

    def __init__(self, cls, window_size=35, columns=None):
        super(AnomalyAdapter, self).__init__(cls)
        self.window_size = window_size
        self.columns = columns

    def calc_anom(self, data):
        if self.columns is None:
            ite = data
        else:
            ite = self.columns
        for column in ite:
            data[column] = calc_anomaly(data[column],
                                        window_size=self.window_size)
        return data

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
    cls : class instance
        Must have a read_ts or read method returning a pandas.DataFrame
    columns: list, optional
        columns in the dataset for which to calculate anomalies.
    kwargs:
        Any additional arguments will be given to the calc_climatology function.
    """

    def __init__(self, cls, columns=None, **kwargs):
        super(AnomalyClimAdapter, self).__init__(cls)
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

    def read_ts(self, *args, **kwargs):
        data = super(AnomalyClimAdapter, self).read_ts(*args, **kwargs)
        return self.calc_anom(data)

    def read(self, *args, **kwargs):
        data = super(AnomalyClimAdapter, self).read(*args, **kwargs)
        return self.calc_anom(data)
