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
Tests for grouping functions
"""

from datetime import date, datetime
from pytesmo.time_series.grouping import grouped_dates_between


def test_grouped_dates_between():
    tstamps_should = [datetime(2007, 1, 10),
                      datetime(2007, 1, 20),
                      datetime(2007, 1, 31),
                      datetime(2007, 2, 10)]
    tstamps = grouped_dates_between(date(2007, 1, 1), date(2007, 2, 1))
    assert tstamps == tstamps_should


def test_grouped_dates_between_start():
    tstamps_should = [datetime(2007, 1, 1),
                      datetime(2007, 1, 11),
                      datetime(2007, 1, 21),
                      datetime(2007, 2, 1)]
    tstamps = grouped_dates_between(
        date(2007, 1, 1), date(2007, 2, 1), start=True)
    assert tstamps == tstamps_should
