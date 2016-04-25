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


class MaskingAdapter(object):
    """
    Transform the given class to return a boolean dataset given the operator
    and threshold. This class calls the callse the read_ts and read methods
    of the given instance and applies boolean masking to the returned data
    using the given operator and threshold.

    Parameters
    ----------
    cls: object
        has to have read_ts method
    operator: string
        one of '<', '<=', '==', '>=', '>'
    threshold: float
        value to use as the threshold combined with the operator

    """

    def __init__(self, cls, op, threshold):
        self.cls = cls

        self.op_lookup = {'<': operator.lt,
                          '<=': operator.le,
                          '==': operator.eq,
                          '>=': operator.ge,
                          '>': operator.gt}
        self.operator = self.op_lookup[op]
        self.threshold = threshold

    def read_ts(self, *args, **kwargs):
        data = self.cls.read_ts(*args, **kwargs)
        return self.operator(data, self.threshold)

    def read(self, *args, **kwargs):
        data = self.cls.read(*args, **kwargs)
        return self.operator(data, self.threshold)


class AnomalyAdapter(object):
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
        self.cls = cls
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
        data = self.cls.read_ts(*args, **kwargs)
        return self.calc_anom(data)

    def read(self, *args, **kwargs):
        data = self.cls.read(*args, **kwargs)
        return self.calc_anom(data)


class AnomalyClimAdapter(object):
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
        self.cls = cls
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
        data = self.cls.read_ts(*args, **kwargs)
        return self.calc_anom(data)

    def read(self, *args, **kwargs):
        data = self.cls.read(*args, **kwargs)
        return self.calc_anom(data)
