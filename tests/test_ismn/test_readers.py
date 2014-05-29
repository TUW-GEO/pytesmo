# Copyright (c) 2013,Vienna University of Technology, Department of Geodesy and Geoinformation
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of the <organization> nor the
#      names of its contributors may be used to endorse or promote products
#      derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''
Created on Jul 31, 2013

@author: Christoph Paulik christoph.paulik@geo.tuwien.ac.at
'''

import os
import unittest
from pytesmo.io.ismn import readers

import pandas as pd
from datetime import datetime
import numpy as np


class TestReaders(unittest.TestCase):

    def setUp(self):
        self.filename_format_header_values = os.path.join(os.path.dirname(__file__), 'test_data', 'format_header_values', 'SMOSMANIA',
                                             'SMOSMANIA_SMOSMANIA_Narbonne_sm_0.050000_0.050000_ThetaProbe-ML2X_20070101_20070131.stm')
        self.filename_format_ceop_sep = os.path.join(os.path.dirname(__file__), 'test_data', 'format_ceop_sep', 'SMOSMANIA',
                                             'SMOSMANIA_SMOSMANIA_Narbonne_sm_0.050000_0.050000_ThetaProbe-ML2X_20070101_20070131.stm')
        self.filename_format_ceop = os.path.join(os.path.dirname(__file__), 'test_data', 'format_ceop', 'SMOSMANIA',
                                             'SMOSMANIA_SMOSMANIA_NBN_20100304_20130801.stm')
        self.filename_malformed = os.path.join(os.path.dirname(__file__), 'test_data', 'malformed',
                                             'mal_formed_file.txt')

        self.metadata_ref = {'network': 'SMOSMANIA',
                            'station': 'Narbonne',
                            'latitude': 43.15,
                            'longitude': 2.9567,
                            'elevation': 112.0,
                            'depth_from': [0.05],
                            'depth_to': [0.05],
                            'variable': ['soil moisture'],
                            'sensor': 'ThetaProbe-ML2X'}

        self.metadata_ref_ceop = dict(self.metadata_ref)
        self.metadata_ref_ceop['depth_from'] = ['multiple']
        self.metadata_ref_ceop['depth_to'] = ['multiple']
        self.metadata_ref_ceop['variable'] = ['ts', 'sm']
        self.metadata_ref_ceop['sensor'] = 'n.s'

    def test_get_info_from_file(self):

        header_elements, filename_elements = readers.get_info_from_file(self.filename_format_ceop_sep)

        assert sorted(header_elements) == sorted(['2007/01/01', '01:00', '2007/01/01',
                                                   '01:00', 'SMOSMANIA', 'SMOSMANIA',
                                                   'Narbonne', '43.15000', '2.95670',
                                                   '112.00', '0.05', '0.05', '0.2140', 'U', 'M'])
        assert sorted(filename_elements) == sorted(['SMOSMANIA', 'SMOSMANIA', 'Narbonne', 'sm',
                                                    '0.050000', '0.050000', 'ThetaProbe-ML2X',
                                                    '20070101', '20070131.stm'])

    def test_get_metadata_header_values(self):

        metadata = readers.get_metadata_header_values(self.filename_format_header_values)

        for key in metadata:
            assert metadata[key] == self.metadata_ref[key]

    def test_reader_format_header_values(self):
        dataset = readers.read_format_header_values(self.filename_format_header_values)
        assert dataset.network == 'SMOSMANIA'
        assert dataset.station == 'Narbonne'
        assert dataset.latitude == 43.15
        assert dataset.longitude == 2.9567
        assert dataset.elevation == 112.0
        assert dataset.variable == ['soil moisture']
        assert dataset.depth_from == [0.05]
        assert dataset.depth_to == [0.05]
        assert dataset.sensor == 'ThetaProbe-ML2X'
        assert type(dataset.data) == pd.DataFrame
        assert dataset.data.index[7] == datetime(2007, 1, 1, 8, 0, 0)
        assert sorted(dataset.data.columns) == sorted(['soil moisture', 'soil moisture_flag', 'soil moisture_orig_flag'])
        assert dataset.data['soil moisture'].values[8] == 0.2135
        assert dataset.data['soil moisture_flag'].values[8] == 'U'
        assert dataset.data['soil moisture_orig_flag'].values[8] == 'M'

    def test_get_metadata_ceop_sep(self):

        metadata = readers.get_metadata_ceop_sep(self.filename_format_ceop_sep)
        for key in metadata:
            assert metadata[key] == self.metadata_ref[key]

    def test_reader_format_ceop_sep(self):
        dataset = readers.read_format_ceop_sep(self.filename_format_ceop_sep)
        assert dataset.network == 'SMOSMANIA'
        assert dataset.station == 'Narbonne'
        assert dataset.latitude == 43.15
        assert dataset.longitude == 2.9567
        assert dataset.elevation == 112.0
        assert dataset.variable == ['soil moisture']
        assert dataset.depth_from == [0.05]
        assert dataset.depth_to == [0.05]
        assert dataset.sensor == 'ThetaProbe-ML2X'
        assert type(dataset.data) == pd.DataFrame
        assert dataset.data.index[7] == datetime(2007, 1, 1, 8, 0, 0)
        assert sorted(dataset.data.columns) == sorted(['soil moisture', 'soil moisture_flag', 'soil moisture_orig_flag'])
        assert dataset.data['soil moisture'].values[8] == 0.2135
        assert dataset.data['soil moisture_flag'].values[8] == 'U'
        assert dataset.data['soil moisture_orig_flag'].values[347] == 'M'

    def test_get_metadata_ceop(self):

        metadata = readers.get_metadata_ceop(self.filename_format_ceop)

        assert metadata == self.metadata_ref_ceop

    def test_reader_format_ceop(self):
        dataset = readers.read_format_ceop(self.filename_format_ceop)
        assert dataset.network == 'SMOSMANIA'
        assert dataset.station == 'Narbonne'
        assert dataset.latitude == 43.15
        assert dataset.longitude == 2.9567
        assert dataset.elevation == 112.0
        assert sorted(dataset.variable) == sorted(['sm', 'ts'])
        assert sorted(dataset.depth_from) == sorted([0.05, 0.1, 0.2, 0.3])
        assert sorted(dataset.depth_to) == sorted([0.05, 0.1, 0.2, 0.3])
        assert dataset.sensor == 'n.s'
        assert type(dataset.data) == pd.DataFrame
        assert dataset.data.index[7] == (0.05, 0.05, datetime(2010, 10, 21, 9, 0, 0))
        assert sorted(dataset.data.columns) == sorted(['sm', 'sm_flag', 'ts', 'ts_flag'])
        assert dataset.data['sm'].values[8] == 0.2227
        assert dataset.data['sm_flag'].values[8] == 'U'
        assert np.isnan(dataset.data.ix[0.3, 0.3]['ts'].values[6])
        assert dataset.data.ix[0.3, 0.3]['ts_flag'].values[6] == 'M'

    def test_reader_get_format(self):
        fileformat = readers.get_format(self.filename_format_header_values)
        assert fileformat == 'header_values'
        fileformat = readers.get_format(self.filename_format_ceop_sep)
        assert fileformat == 'ceop_sep'
        fileformat = readers.get_format(self.filename_format_ceop)
        assert fileformat == 'ceop'
        with self.assertRaises(readers.ReaderException):
            fileformat = readers.get_format(self.filename_malformed)

if __name__ == '__main__':
    unittest.main()
