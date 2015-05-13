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
Created on Sep 24, 2013

@author: Christoph.Paulik@geo.tuwien.ac.at
'''

import pytesmo.temporal_matching as temp_match

import pandas as pd


class BasicTemporalMatching(object):
    """
    Temporal matching object

    Parameters
    ----------
    window : float
        window size to use for temporal matching. A match in other will only
        be found if it is +- window size days away from a point in reference
    """

    def __init__(self, window=0.5, reverse=False):
        self.window = window

        if reverse:
            self.match = self.match_reverse

    def match(self, reference, other):
        """
        takes reference and other dataframe and returnes a joined Dataframe
        in this case the reference dataset for the grid is also the
        temporal reference dataset
        """
        # temporal match comparison to reference TimeSeries
        try:
            matched_other = temp_match.df_match(reference, other,
                                                window=self.window, dropna=True)
        except ValueError:
            return pd.DataFrame()
        matched_other = matched_other.drop(['distance', 'index'], axis=1)

        return matched_other.join(reference)

    def match_reverse(self, reference, other):
        """
        takes reference and other dataframe and returnes a joined Dataframe
        in this case the reference dataset for the grid is also the
        temporal reference dataset
        """
        # temporal match comparison to reference TimeSeries
        try:
            matched_ref = temp_match.df_match(other, reference,
                                              window=self.window, dropna=True)
        except ValueError:
            return pd.DataFrame()
        matched_ref = matched_ref.drop(['distance', 'index'], axis=1)

        return matched_ref.join(other)
