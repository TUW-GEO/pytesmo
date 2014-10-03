# Copyright (c) 2014, Vienna University of Technology (TU Wien), Department
# of Geodesy and Geoinformation (GEO).
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of the Vienna University of Technology,
#      Department of Geodesy and Geoinformation nor the
#      names of its contributors may be used to endorse or promote products
#      derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL VIENNA UNIVERSITY OF TECHNOLOGY,
# DEPARTMENT OF GEODESY AND GEOINFORMATION BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Author: Christoph Paulik christoph.paulik@geo.tuwien.ac.at
# Creation date: 2014-06-30


"""
Module provides grouping functions that can be used together with pandas
to create a few strange timegroupings like e.g. decadal products were
there are three products per month with timestamps on the 10th 20th and last
of the month
"""

import pandas as pd
import numpy as np
from datetime import date
import calendar


def group_by_day_bin(df, bins=[1, 11, 21, 32], start=False,
                     dtindex=None):
    """
    Calculates timegroups for a given daterange. Groups are from day
    1-10,
    11-20,
    21-last day of each month.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with DateTimeIndex for which the grouping should be
        done
    bins : list, optional
        bins in day of the month, default is for dekadal grouping
    start : boolean, optional
        if set to True the start of the bin will be the timestamp for each
        observations
    dtindex : pandas.DatetimeIndex, optional
        precomputed DatetimeIndex that should be used for resulting
        groups, useful for processing of numerous datasets since it does not
        have to be computed for every call

    Returns
    -------
    grouped : pandas.core.groupby.DataFrameGroupBy
        DataFrame groupby object according the the day bins
        on this object functions like sum() or mean() can be
        called to get the desired aggregation.
    dtindex : pandas.DatetimeIndex
        returned so that it can be reused if possible
    """

    dekads = np.digitize(df.index.day, bins)
    if dtindex is None:
        dtindex = grp_to_datetimeindex(dekads, bins, df.index, start=start)
    grp = pd.DataFrame(df.as_matrix(), columns=df.columns, index=dtindex)
    return grp.groupby(level=0), dtindex


def grp_to_datetimeindex(grps, bins, dtindex, start=False):
    """Makes a datetimeindex that has for each entry the timestamp
    of the bin beginning or end this entry belongs to.

    Parameters
    ----------
    grps : numpy.array
        group numbers made by np.digitize(data, bins)
    bins : list
        bin start values e.g. [0,11,21] would be two bins one with
        values 0<=x<11 and the second one with 11<=x<21
    dtindex : pandas.DatetimeIndex
        same length as grps, gives the basis datetime for each group
    start : boolean, optional
        if set to True the start of the bin will be the timestamp for each
        observations

    Returns
    -------
    grpdt : pd.DatetimeIndex
        Datetimeindex where every date is the end of the bin the datetime
        ind the input dtindex belongs to
    """

    dtlist = []
    offset = 1
    index_offset = 0
    # select previous bin and do not subtract a day if start is set to True
    if start:
        offset = 0
        index_offset = -1

    for i, el in enumerate(dtindex):
        _, max_day_month = calendar.monthrange(el.year, el.month)

        dtlist.append(date(el.year, el.month, min([bins[grps[i] + index_offset]
                                                   - offset, max_day_month])))

    return pd.DatetimeIndex(dtlist)


def grouped_dates_between(start_date, end_date, bins=[1, 11, 21, 32], start=False):
    """
    Between a start and end date give all dates that represent a bin
    See test for example.

    Parameters
    ----------
    start_date: date
        start date
    end_date: date
        end date
    bins: list, optional
        bin start values as days in a month e.g. [0,11,21] would be two bins one with
        values 0<=x<11 and the second one with 11<=x<21
    start: boolean, optional
        if True the start of the bins is the representative date

    Returns
    -------
    tstamps : list of datetimes
        list of representative dates between start and end date
    """

    daily = pd.date_range(start_date, end_date, freq='D')
    fake_data = pd.DataFrame(np.arange(len(daily)), index=daily)
    grp, dtindex = group_by_day_bin(fake_data, bins=bins, start=start)
    tstamps = grp.sum().index.to_pydatetime().tolist()

    return tstamps
