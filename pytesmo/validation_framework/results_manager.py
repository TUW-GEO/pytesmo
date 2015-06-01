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
Created on Nov 11, 2013

These classes take the result lists produced by a metrics calculator
and save them on the given grid

@author: Christoph Paulik christoph.paulik@geo.tuwien.ac.at
'''

from netCDF4 import Dataset
import numpy as np


class NetCDFManager(object):

    def __init__(self, metrics_calculator, grid,
                 gpi_field='gpi'):

        self.metrics_calc = metrics_calculator
        self.grid = grid
        self.gpi_field = gpi_field
        self.gpi_nan_value = self.metrics_calc.result_template[self.gpi_field]
        self.save_fields = self.metrics_calc.result_template.dtype.fields

    def manage_results(self, result_dict, save_file):

        ncfile = Dataset(save_file, 'w')
        dim = ncfile.createDimension(
            self.gpi_field, size=len(self.grid.activegpis))

        for key in result_dict:
            data = result_dict[key]

            index = np.where((data[self.gpi_field] != self.gpi_nan_value))
            valid_gpis = data[self.gpi_field][index]
            data_template = self.metrics_calc.result_template.copy()
            result_template = data_template.repeat(self.grid.n_gpi)
            result_template[valid_gpis] = data[index]
            if self.grid.subset is None:
                save_data = result_template
            else:
                save_data = result_template[self.grid.subset]

            for field in self.save_fields:
                var = ncfile.createVariable("%s_%s" % (key, field),
                                            save_data[field].dtype, self.gpi_field)
                var[:] = save_data[field]

        ncfile.close()
