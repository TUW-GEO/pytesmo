# Copyright (c) 2013,Vienna University of Technology, Department of Geodesy and Geoinformation
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of the Vienna University of Technology, Department of Geodesy and Geoinformation nor the
#      names of its contributors may be used to endorse or promote products
#      derived from this software without specific prior written permission.

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

'''
Created on Oct 16, 2013

@author: Christoph Paulik christoph.paulik@geo.tuwien.ac.at
'''

import pandas as pd
import numpy as np

from pytesmo.timedate.julian import julday
from pytesmo.time_series.filters import boxcar_filter


def moving_average(Ser,
                   window_size=1):
    '''
    Applies a moving average (box) filter on an input time series

    Parameters
    ----------
    Ser : pandas.Series (index must be a DateTimeIndex or julian date)

    window_size : float, optional
        The size of the moving_average window [days] that will be applied on the
        input Series
        Default: 1

    Returns
    -------
    Ser : pandas.Series
        moving-average filtered time series
    '''
    # if index is datetimeindex then convert it to julian date
    if type(Ser.index) == pd.DatetimeIndex:

        jd_index = julday(np.asarray(Ser.index.month),
                          np.asarray(Ser.index.day),
                          np.asarray(Ser.index.year),
                          np.asarray(Ser.index.hour),
                          np.asarray(Ser.index.minute),
                          np.asarray(Ser.index.second))

    else:
        jd_index = Ser.index.values

    filtered = boxcar_filter(
        np.atleast_1d(np.squeeze(Ser.values.astype(np.double))),
        jd_index.astype(np.double),
        window=window_size)

    result = pd.Series(filtered, index=Ser.index)

    return result
