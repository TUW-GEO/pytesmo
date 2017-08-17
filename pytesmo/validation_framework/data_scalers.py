# Copyright (c) 2017,Vienna University of Technology,
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
Data scaler classes to be used together with the validation framework.
'''

import pytesmo.scaling as scaling


class DefaultScaler(object):
    """
    Scaling class that implements the scaling based on a
    given method from the pytesmo.scaling module.

    Parameters
    ----------
    method: string
        The data will be scaled into the reference space using the
        method specified by this string.
    """

    def __init__(self, method):
        self.method = method

    def scale(self, data, reference_index, gpi_info):
        """
        Scale all columns in data to the
        column at the reference_index.

        Parameters
        ----------
        data: pandas.DataFrame
            temporally matched dataset
        reference_index: int
            Which column of the data contains the
            scaling reference.
        gpi_info: tuple
            tuple of at least, (gpi, lon, lat)
            Useful if scaling parameters are e.g. stored.

        Raises
        ------
        ValueError
            if scaling is not successful
        """
        return scaling.scale(data,
                             method=self.method,
                             reference_index=reference_index)
