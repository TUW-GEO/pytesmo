# Copyright (c) 2013,Vienna University of Technology, Department of Geodesy and
# Geoinformation
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of the Vienna University of Technology, Department of
#      Geodesy and Geoinformation nor the names of its contributors may be used
#      to endorse or promote products derived from this software without
#      specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL VIENNA UNIVERSITY OF TECHNOLOGY, DEPARTMENT
# OF GEODESY AND GEOINFORMATION BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''
Created on Sep 24, 2013

@author: Christoph.Paulik@geo.tuwien.ac.at
'''

import itertools
import pandas as pd

import pytesmo.temporal_matching as temporal_matching


class BasicTemporalMatching(object):
    """
    Temporal matching object

    Parameters
    ----------
    window : float
        window size to use for temporal matching. A match in other will only
        be found if it is +- window size days away from a point in reference
    """

    def __init__(self, window=0.5):
        self.window = window

    def match(self, reference, *args):
        """
        takes reference and other dataframe and returnes a joined Dataframe
        in this case the reference dataset for the grid is also the
        temporal reference dataset
        """
        ref_df = pd.DataFrame(reference)
        return temporal_matching.combined_temporal_collocation(
            ref_df, args, self.window, dropna=True, dropduplicates=True,
            add_ref_data=True, combined_dropna="all"
        )

    def combinatory_matcher(self, df_dict, refkey, n=2):
        """
        Basic temporal matcher that matches always one Dataframe to
        the reference Dataframe resulting in matched DataFrame pairs.

        If the input dict has the keys 'data1' and 'data2' then the
        output dict will have the key ('data1', 'data2'). The new key
        is stored as a tuple to avoid any issues with string concetanation.

        During matching the column names of the dataframes will be
        transformed into MultiIndex to ensure unique names.

        Parameters
        ----------
        df_dict: dict of pandas.DataFrames
            dictionary containing the spatially colocated DataFrames.
        refkey: string
            key into the df_dict of the DataFrame that should be taken
            as a reference.
        n: int
            number of datasets to match at once

        Returns
        -------
        matched: dict of pandas.DataFrames
            Dictionary containing matched DataFrames. The key is put
            together from the keys of the input dict as a tuple of the
            keys of the datasets this dataframe contains.
        """
        matched = {}
        keys = list(df_dict)
        keys.pop(keys.index(refkey))
        ref_df = df_dict[refkey]
        ref_df = df_name_multiindex(ref_df, refkey)

        for iterkeys in itertools.combinations(keys, n - 1):
            match_list = []
            match_key = []
            for key in iterkeys:
                other_df = df_dict[key]
                other_df = df_name_multiindex(other_df, key)
                match_list.append(other_df)
                match_key.append(key)

            matched_key = tuple([refkey] + sorted(match_key))
            joined = self.match(ref_df,
                                *match_list)

            if len(joined) != 0:
                matched[matched_key] = joined

        return matched


def dfdict_combined_temporal_collocation(
    dfs, refname, window=None, n=None, **kwargs
):
    """
    Applies :py:func:`combined_temporal_collocation` on a dictionary of
    dataframes.

    Parameters
    ----------
    dfs : dict
        Dictionary of pd.DataFrames containing the dataframes to be collocated.
    refname : str
        Name of the reference frame in `dfs`.
    window : pd.Timedelta or float, optional
        Window around reference timestamps in which to look for data. Floats
        are interpreted as number of days. If it is not given, defaults to 1
        hour to mimick the behaviour of
        ``BasicTemporalMatching.combinatory_matcher``.
    n : dummy argument
        Will be ignored.
    **kwargs :
        Keyword arguments passed to :py:func:`combined_temporal_collocation`.

    Returns:
    --------
    matched_dict : dict
        Dictionary where the keys are tuples of ``(other_name, refname)`` for
        each other key in `dfs`, and values are the matched dataframes.
        The column names of the dataframes are again tuples of ``(name, col)``
        where `name` is the key from `dfs` and `col` is the original column
        name in the input dataframe.
    """
    if window is None:
        window = pd.Timedelta(hours=1)

    others = []
    for name in dfs:
        if name != refname:
            others.append(df_name_multiindex(dfs[name], name))
    ref = df_name_multiindex(dfs[refname], refname)
    matched_df = temporal_matching.combined_temporal_collocation(
        ref, others, window, add_ref_data=True, combined_dropna=True, **kwargs
    )

    # unpack again to dictionary
    matched_dict = {}
    for name in dfs:
        if name != refname:
            tuple_keys = list(itertools.product((name,), dfs[name].columns))
            keep = list(ref.columns) + tuple_keys
            # there's a strange bug in pytesmo, when I make the order here to
            # be (refname, name) instead of (name, refname) I don't get all the
            # results
            matched_dict[(name, refname)] = matched_df[keep]
            # rename_dict = {tk: tk[1] for tk in tuple_keys}
            # matched_dict[(refname, name)].rename(rename_dict, axis=1,
            # inplace=True)
    return matched_dict


def make_combined_temporal_matcher(window):
    """
    Matches multiple dataframes together to only have common timestamps.

    See
    :py:func:`pytesmo.temporal_matching.dfdict_combined_temporal_collocation`
    for more details
    """

    def matcher(dfs, refname, n=None, **kwargs):
        return dfdict_combined_temporal_collocation(dfs, refname, **kwargs)
    return matcher


def df_name_multiindex(df, name):
    """
    Rename columns of a DataFrame by using new column names that
    are tuples of (name, column_name) to ensure unique column names
    that can also be split again. This transforms the columns to a MultiIndex.
    """
    d = {}
    for c in df.columns:
        d[c] = (name, c)

    return df.rename(columns=d)
