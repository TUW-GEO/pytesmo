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

import pandas as pd

class DataAverager():
    """
    Class to handle multiple measurements falling under the same reference gridpoint. The goal is to include
    here all identified upscaling methods to provide an estimate at the reference footprint scale.

    Implemented methods:

        * time-stability filtering
        * simple averaging

    Parameters
    ----------
    ref: pygeogrids Grid obj.
        Class containing the method read_ts for reading the data of the reference
    others: dict
        Dict of shape 'other_name': pygeogrids Grid obj. for the other dataset
    """
    def __init__(
            self,
            ref,
            others,
    ):
        self.ref = ref_class
        self.others = others_class

    @property
    def lut(self) -> dict:
        """Get a lookup table that combines the points falling under the same reference pixel"""
        lut = {}
        for other_name, other_class in self.others.items():
            other_points = other_class.grid.get_grid_points()
            other_lut = {}
            # iterate from the side of the non-reference
            for point in other_points:
                gpi, lon, lat = point
                # list all non-ref points under the same ref gpi
                ref_gpi = self.ref_class.find_nearest_gpi(lon, lat)[0]
                if ref_gpi in other_lut.keys():
                    other_lut[ref_gpi].append(point)
                else:
                    other_lut[ref_gpi] = [point]
            # add to dictionary even when empty
            lut[other_name] = other_lut

        return lut




