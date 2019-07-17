
Why is temporal matching important?
-----------------------------------

Satellite observations usually have an irregular temporal sampling
pattern (intervals between 6-36 hours), which is mostly controlled by
the orbit of the satellite and the instrument measurement geometry. On
the other hand, in-situ instruments or land surface models generally
sample on regular time intervals (commonly every 1, 3, 6, 12 or 24
hours). In order to compute error/performance statistics (such as RMSD,
bias, correlation) between the time series coming different sources, it
is required that observation pairs (or triplets, etc.) are found which
(nearly) coincide in time. A simple way to identify such pairs is by
using a nearest neighbor search. First, one time series needs to be
selected as temporal reference (i.e. all other time series will be
matched to this reference) and second, a tolerance window (typically
around 1-12 hours) has to be defined characterizing the temporal
correlation of neighboring observation (i.e. observations outside of the
tolerance window are no longer be considered as representative
neighbors). An important special case may occur during the nearest
neighbor search, which leads to duplicated neighbors. Depending on the
application and use-case, the user needs to decide whether to keep the
duplicates or to remove them before computing any statistics.

Matching two time series
------------------------

The following examples shows how to match two time series with regular
and irregular temporal sampling.

.. code:: ipython3

    import numpy as np
    import pandas as pd
    
    from pytesmo.temporal_matching import df_match
    
    # create reference time series as dataframe
    ref_index = pd.date_range("2007-01-01", "2007-01-05", freq="D")
    ref_data = np.arange(len(ref_index))
    ref_df = pd.DataFrame({"data": ref_data}, index=ref_index)
    
    # create other time series as dataframe
    match_index = pd.date_range("2007-01-01 09:00:00", "2007-01-05 09:00:00", freq="D")
    match_data = np.arange(len(match_index))
    match_df = pd.DataFrame({"matched_data": match_data}, index=match_index)
    
    # match time series
    matched = df_match(ref_df, match_df)
    
    # test if data and index are correct
    print(matched)
    np.testing.assert_allclose(5 * [9/24.], matched.distance.values)
    np.testing.assert_allclose(np.arange(5), matched.matched_data)


.. parsed-literal::

                distance  matched_data               index
    2007-01-01     0.375             0 2007-01-01 09:00:00
    2007-01-02     0.375             1 2007-01-02 09:00:00
    2007-01-03     0.375             2 2007-01-03 09:00:00
    2007-01-04     0.375             3 2007-01-04 09:00:00
    2007-01-05     0.375             4 2007-01-05 09:00:00


.. code:: ipython3

    # create other (irregular) time series as dataframe
    match_irr_index = pd.to_datetime(["2007-01-01 04:00:00", "2007-01-01 22:00:00", 
                                      "2007-01-02 06:00:00", "2007-01-03 12:00:00"])
    match_irr_data = np.arange(len(match_irr_index))
    match_irr_df = pd.DataFrame({"matched_data": match_irr_data}, index=match_irr_index)
    
    # match time series with 8 hour time window
    matched = df_match(ref_df, match_irr_df, window=8/24.)
    
    # test if data and index are correct
    print(matched)
    np.testing.assert_allclose([4/24., -2/24., np.nan, np.nan, np.nan], matched.distance.values)
    np.testing.assert_allclose([0, 1, np.nan, np.nan, np.nan], matched.matched_data)


.. parsed-literal::

                distance  matched_data               index
    2007-01-01  0.166667           0.0 2007-01-01 04:00:00
    2007-01-02 -0.083333           1.0 2007-01-01 22:00:00
    2007-01-03       NaN           NaN                 NaT
    2007-01-04       NaN           NaN                 NaT
    2007-01-05       NaN           NaN                 NaT


.. code:: ipython3

    # match time series with 8 hour time window and drop nan
    matched = df_match(ref_df, match_irr_df, window=8/24., dropna=True)
    
    # test if data and index are correct
    print(matched)
    np.testing.assert_allclose([4/24., -2/24.], matched.distance.values)
    np.testing.assert_allclose([0, 1], matched.matched_data)


.. parsed-literal::

                distance  matched_data               index
    2007-01-01  0.166667           0.0 2007-01-01 04:00:00
    2007-01-02 -0.083333           1.0 2007-01-01 22:00:00


Special case of duplicated neighbor
-----------------------------------

.. code:: ipython3

    # create reference time series as dataframe
    ref_index = pd.to_datetime(["2007-01-01 04:00:00", "2007-01-01 06:00:00", 
                                "2007-01-02 06:00:00", "2007-01-02 08:00:00"])
    ref_data = np.arange(len(ref_index))
    ref_df = pd.DataFrame({"data": ref_data}, index=ref_index)
    
    # create other time series as dataframe
    ref_index = pd.date_range("2007-01-01 00:00:00", "2007-01-05 00:00:00", freq="3h")
    match_data = np.arange(len(match_index))
    match_df = pd.DataFrame({"matched_data": match_data}, index=match_index)
    
    # match time series
    matched = df_match(ref_df, match_df)
    
    print(matched)


.. parsed-literal::

                         distance  matched_data               index
    2007-01-01 04:00:00  0.208333             0 2007-01-01 09:00:00
    2007-01-01 06:00:00  0.125000             0 2007-01-01 09:00:00
    2007-01-02 06:00:00  0.125000             1 2007-01-02 09:00:00
    2007-01-02 08:00:00  0.041667             1 2007-01-02 09:00:00


.. code:: ipython3

    # match time series and drop duplicates
    matched = df_match(ref_df, match_df, dropduplicates=True)
    
    print(matched)


.. parsed-literal::

                         distance  matched_data               index
    2007-01-01 06:00:00  0.125000             0 2007-01-01 09:00:00
    2007-01-02 08:00:00  0.041667             1 2007-01-02 09:00:00


Matching three or more time series
----------------------------------

.. code:: ipython3

    # create reference time series as dataframe
    ref_index = pd.to_datetime(["2007-01-01 04:00:00", "2007-01-01 06:00:00", 
                                "2007-01-02 06:00:00", "2007-01-02 08:00:00",
                                "2007-01-03 09:00:00", "2007-01-03 10:00:00"])
    ref_data = np.arange(len(ref_index))
    ref_df = pd.DataFrame({"data": ref_data}, index=ref_index)
    
    # create other time series as dataframe
    match_index = pd.date_range("2007-01-01 00:00:00", "2007-01-05 00:00:00", freq="3h")
    match_data = np.arange(len(match_index))
    match_df1 = pd.DataFrame({"matched_data": match_data}, index=match_index)
    
    # create other time series as dataframe
    match_index = pd.date_range("2007-01-01 00:00:00", "2007-01-05 00:00:00", freq="6h")
    match_data = np.arange(len(match_index))
    match_df2 = pd.DataFrame({"matched_data": match_data}, index=match_index)
    
    # match time series
    matched = df_match(ref_df, match_df1, match_df2)
    
    print(matched[0])
    
    print(matched[1])


.. parsed-literal::

                         distance  matched_data               index
    2007-01-01 04:00:00 -0.041667             1 2007-01-01 03:00:00
    2007-01-01 06:00:00  0.000000             2 2007-01-01 06:00:00
    2007-01-02 06:00:00  0.000000            10 2007-01-02 06:00:00
    2007-01-02 08:00:00  0.041667            11 2007-01-02 09:00:00
    2007-01-03 09:00:00  0.000000            19 2007-01-03 09:00:00
    2007-01-03 10:00:00 -0.041667            19 2007-01-03 09:00:00
                         distance  matched_data               index
    2007-01-01 04:00:00  0.083333             1 2007-01-01 06:00:00
    2007-01-01 06:00:00  0.000000             1 2007-01-01 06:00:00
    2007-01-02 06:00:00  0.000000             5 2007-01-02 06:00:00
    2007-01-02 08:00:00 -0.083333             5 2007-01-02 06:00:00
    2007-01-03 09:00:00 -0.125000             9 2007-01-03 06:00:00
    2007-01-03 10:00:00  0.083333            10 2007-01-03 12:00:00


.. code:: ipython3

    # match time series and drop duplicates
    matched = df_match(ref_df, match_df1, match_df2, dropduplicates=True)
    
    print(matched[0])
    
    print(matched[1])


.. parsed-literal::

                         distance  matched_data               index
    2007-01-01 04:00:00 -0.041667             1 2007-01-01 03:00:00
    2007-01-01 06:00:00  0.000000             2 2007-01-01 06:00:00
    2007-01-02 06:00:00  0.000000            10 2007-01-02 06:00:00
    2007-01-02 08:00:00  0.041667            11 2007-01-02 09:00:00
    2007-01-03 09:00:00  0.000000            19 2007-01-03 09:00:00
                         distance  matched_data               index
    2007-01-01 06:00:00  0.000000             1 2007-01-01 06:00:00
    2007-01-02 06:00:00  0.000000             5 2007-01-02 06:00:00
    2007-01-03 09:00:00 -0.125000             9 2007-01-03 06:00:00
    2007-01-03 10:00:00  0.083333            10 2007-01-03 12:00:00


.. code:: ipython3

    # match time series, 2 hour window and drop duplicates
    matched = df_match(ref_df, match_df1, match_df2, window=2/24., dropduplicates=True)
    
    print(matched[0])
    
    print(matched[1])


.. parsed-literal::

                         distance  matched_data               index
    2007-01-01 04:00:00 -0.041667           1.0 2007-01-01 03:00:00
    2007-01-01 06:00:00  0.000000           2.0 2007-01-01 06:00:00
    2007-01-02 06:00:00  0.000000          10.0 2007-01-02 06:00:00
    2007-01-02 08:00:00  0.041667          11.0 2007-01-02 09:00:00
    2007-01-03 09:00:00  0.000000          19.0 2007-01-03 09:00:00
                         distance  matched_data               index
    2007-01-01 06:00:00  0.000000           1.0 2007-01-01 06:00:00
    2007-01-02 06:00:00  0.000000           5.0 2007-01-02 06:00:00
    2007-01-03 10:00:00  0.083333          10.0 2007-01-03 12:00:00




