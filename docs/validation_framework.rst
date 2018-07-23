
The pytesmo validation framework
================================

The pytesmo validation framework takes care of iterating over datasets,
spatial and temporal matching as well as scaling. It uses metric
calculators to then calculate metrics that are returned to the user.
There are several metrics calculators included in pytesmo but new ones
can be added simply.

Overview
--------

How does the validation framework work? It makes these assumptions about
the used datasets:

-  The dataset readers that are used have a ``read_ts`` method that can
   be called either by a grid point index (gpi) which can be any
   indicator that identifies a certain grid point or by using longitude
   and latitude. This means that both call signatures ``read_ts(gpi)``
   and ``read_ts(lon, lat)`` must be valid. Please check the
   `pygeobase <https://github.com/TUW-GEO/pygeobase>`__ documentation
   for more details on how a fully compatible dataset class should look.
   But a simple ``read_ts`` method should do for the validation
   framework. This assumption can be relaxed by using the
   ``read_ts_names`` keyword in the
   pytesmo.validation\_framework.data\_manager.DataManager class.
-  The ``read_ts`` method returns a pandas.DataFrame time series.
-  Ideally the datasets classes also have a ``grid`` attribute that is a
   `pygeogrids <http://pygeogrids.readthedocs.org/en/latest/>`__ grid.
   This makes the calculation of lookup tables easily possible and the
   nearest neighbor search faster.

Fortunately these assumptions are true about the dataset readers
included in pytesmo.

It also makes a few assumptions about how to perform a validation. For a
comparison study it is often necessary to choose a spatial reference
grid, a temporal reference and a scaling or data space reference.

Spatial reference
~~~~~~~~~~~~~~~~~

The spatial reference is the one to which all the other datasets are
matched spatially. Often through nearest neighbor search. The validation
framework uses grid points of the dataset specified as the spatial
reference to spatially match all the other datasets with nearest
neighbor search. Other, more sophisticated spatial matching algorithms
are not implemented at the moment. If you need a more complex spatial
matching then a preprocessing of the data is the only option at the
moment.

Temporal reference
~~~~~~~~~~~~~~~~~~

The temporal reference is the dataset to which the other dataset are
temporally matched. That means that the nearest observation to the
reference timestamps in a certain time window is chosen for each
comparison dataset. This is by default done by the temporal matching
module included in pytesmo. How many datasets should be matched to the
reference dataset at once can be configured, we will cover how to do
this later.

Data space reference
~~~~~~~~~~~~~~~~~~~~

It is often necessary to bring all the datasets into a common data space
by using. Scaling is often used for that and pytesmo offers a choice of
several scaling algorithms (e.g. CDF matching, min-max scaling, mean-std
scaling, triple collocation based scaling). The data space reference can
also be chosen independently from the other two references.

Data Flow
---------

After it is initialized, the validation framework works through the
following steps:

1. Read all the datasets for a certain job (gpi, lon, lat)
2. Read all the masking dataset if any
3. Mask the temporal reference dataset using the masking data
4. Temporally match all the chosen combinations of temporal reference
   and other datasets
5. Turn the temporally matched time series over to the metric
   calculators
6. Get the calculated metrics from the metric calculators
7. Put all the metrics into a dictionary by dataset combination and
   return them.

Masking datasets
----------------

Masking datasets can be used if the datasets that are compared do not
contain the necessary information to mask them. For example we might
want to use modelled soil temperature data to mask our soil moisture
observations before comparing them. To be able to do that we just need a
Dataset that returns a pandas.DataFrame with one column of boolean data
type. Everywhere where the masking dataset is ``True`` the data will be
masked.

Let's look at a first example.

Example soil moisture validation: ASCAT - ISMN
----------------------------------------------

This example shows how to setup the pytesmo validation framework to
perform a comparison between ASCAT and ISMN data.

.. code:: ipython2

    import os
    
    import pytesmo.validation_framework.metric_calculators as metrics_calculators
    
    from datetime import datetime
    
    from ascat.timeseries import AscatSsmCdr
    from pytesmo.io.ismn.interface import ISMN_Interface
    from pytesmo.validation_framework.validation import Validation
    from pytesmo.validation_framework.results_manager import netcdf_results_manager

You need the test data from https://github.com/TUW-GEO/pytesmo-test-data
for this example

.. code:: ipython2

    testdata_folder = '/pytesmo/testdata'
    output_folder = '/pytesmo/code/examples/output'

First we initialize the data readers that we want to use. In this case
the ASCAT soil moisture time series and in situ data from the ISMN.

Initialize ASCAT reader

.. code:: ipython2

    ascat_data_folder = os.path.join(testdata_folder,
                                     'sat/ascat/netcdf/55R22')
    ascat_grid_folder = os.path.join(testdata_folder,
                                     'sat/ascat/netcdf/grid')
    static_layers_folder = os.path.join(testdata_folder,
                                        'sat/h_saf/static_layer')
    
    ascat_reader = AscatSsmCdr(ascat_data_folder, ascat_grid_folder,
                               grid_filename='TUW_WARP5_grid_info_2_1.nc',
                               static_layer_path=static_layers_folder)
    ascat_reader.read_bulk = True


.. parsed-literal::

    /space/tools/miniconda3/envs/pytesmo/lib/python2.7/site-packages/ascat/timeseries.py:128: UserWarning: WARNING: valid_range not used since it
    cannot be safely cast to variable data type
      land_gp = np.where(grid_nc.variables['land_flag'][:] == 1)[0]


Initialize ISMN reader

.. code:: ipython2

    ismn_data_folder = os.path.join(testdata_folder,
                                     'ismn/multinetwork/header_values')
    
    ismn_reader = ISMN_Interface(ismn_data_folder)

The validation is run based on jobs. A job consists of at least three
lists or numpy arrays specifing the grid point index, its latitude and
longitude. In the case of the ISMN we can use the ``dataset_ids`` that
identify every time series in the downloaded ISMN data as our grid point
index. We can then get longitude and latitude from the metadata of the
dataset.

**DO NOT CHANGE** the name ***jobs*** because it will be searched during
the parallel processing!

.. code:: ipython2

    jobs = []
    
    ids = ismn_reader.get_dataset_ids(variable='soil moisture', min_depth=0, max_depth=0.1)
    for idx in ids:
        metadata = ismn_reader.metadata[idx]
        jobs.append((idx, metadata['longitude'], metadata['latitude']))
    
    print("Jobs (gpi, lon, lat):")
    print(jobs)


.. parsed-literal::

    Jobs (gpi, lon, lat):
    [(0, -120.78559, 38.14956), (1, -120.9675, 38.43003), (2, -120.80639, 38.17353), (3, -86.55, 34.783), (4, -97.083, 37.133), (5, -105.417, 34.25), (6, 102.1333, 33.8833), (7, 102.1333, 33.6666)]


For this small test dataset it is only one job

It is important here that the ISMN reader has a read\_ts function that
works by just using the ``dataset_id``. In this way the validation
framework can go through the jobs and read the correct time series.

.. code:: ipython2

    data = ismn_reader.read_ts(ids[0])
    print('ISMN data example:')
    print(data.head())


.. parsed-literal::

    ISMN data example:
                         soil moisture soil moisture_flag  soil moisture_orig_flag
    date_time                                                                     
    2012-12-14 19:00:00         0.3166                  U                        0
    2012-12-14 20:00:00         0.3259                  U                        0
    2012-12-14 21:00:00         0.3259                  U                        0
    2012-12-14 22:00:00         0.3263                  U                        0
    2012-12-14 23:00:00         0.3263                  U                        0


Initialize the Validation class
-------------------------------

The Validation class is the heart of the validation framwork. It
contains the information about which datasets to read using which
arguments or keywords and if they are spatially compatible. It also
contains the settings about which metric calculators to use and how to
perform the scaling into the reference data space. It is initialized in
the following way:

.. code:: ipython2

    datasets = {
        'ISMN': {
            'class': ismn_reader,
            'columns': ['soil moisture']
        },
        'ASCAT': {
            'class': ascat_reader,
            'columns': ['sm'],
            'kwargs': {'mask_frozen_prob': 80,
                       'mask_snow_prob': 80,
                       'mask_ssf': True}
        }}

The datasets dictionary contains all the information about the datasets
to read. The ``class`` is the dataset class to use which we have already
initialized. The ``columns`` key describes which columns of the dataset
interest us for validation. This a mandatory field telling the framework
which other columns to ignore. In this case the columns
``soil moisture_flag`` and ``soil moisture_orig_flag`` will be ignored
by the ISMN reader. We can also specify additional keywords that should
be given to the ``read_ts`` method of the dataset reader. In this case
we want the ASCAT reader to mask the ASCAT soil moisture using the
included frozen and snow probabilities as well as the SSF. There are
also other keys that can be used here. Please see the documentation for
explanations.

.. code:: ipython2

    period = [datetime(2007, 1, 1), datetime(2014, 12, 31)]
    basic_metrics = metrics_calculators.BasicMetrics(other_name='k1')
    
    process = Validation(
        datasets, 'ISMN',
        temporal_ref='ASCAT',
        scaling='lin_cdf_match',
        scaling_ref='ASCAT',   
        metrics_calculators={(2, 2): basic_metrics.calc_metrics},
        period=period)

During the initialization of the Validation class we can also tell it
other things that it needs to know. In this case it uses the datasets we
have specified earlier. The spatial reference is the ``'ISMN'`` dataset
which is the second argument. The 'metrics\_calculators' argument looks
a little bit strange so let's look at it in more detail.

It is a dictionary with a tuple as the key and a function as the value.
The key tuple ``(n, k)`` has the following meaning: ``n`` datasets are
temporally matched together and then given in sets of ``k`` columns to
the metric calculator. The metric calculator then gets a DataFrame with
the columns ['ref', 'k1', 'k2' ...] and so on depending on the value of
k. The value of ``(2, 2)`` makes sense here since we only have two
datasets and all our metrics also take two inputs.

This can be used in more complex scenarios to e.g. have three input
datasets that are all temporally matched together and then combinations
of two input datasets are given to one metric calculator while all three
datasets are given to another metric calculator. This could look like
this:

.. code:: python

    { (3 ,2): metric_calc,
      (3, 3): triple_collocation}

Create the variable ***save\_path*** which is a string representing the
path where the results will be saved. **DO NOT CHANGE** the name
***save\_path*** because it will be searched during the parallel
processing!

.. code:: ipython2

    save_path = output_folder
    
    import pprint
    for job in jobs:
        
        results = process.calc(*job)
        pprint.pprint(results)
        netcdf_results_manager(results, save_path)


.. parsed-literal::

    /space/tools/miniconda3/envs/pytesmo/lib/python2.7/site-packages/pandas/core/reshape/merge.py:558: UserWarning: merging between different levels can give an unintended result (1 levels on the left, 2 on the right)
      warnings.warn(msg, UserWarning)
    /space/tools/miniconda3/envs/pytesmo/lib/python2.7/site-packages/pandas/core/reshape/merge.py:558: UserWarning: merging between different levels can give an unintended result (2 levels on the left, 1 on the right)
      warnings.warn(msg, UserWarning)


.. parsed-literal::

    {(('ASCAT', 'sm'), ('ISMN', 'soil moisture')): {'BIAS': array([-1.9682411], dtype=float32),
                                                    'R': array([0.79960084], dtype=float32),
                                                    'RMSD': array([13.0622425], dtype=float32),
                                                    'gpi': array([0], dtype=int32),
                                                    'lat': array([38.14956]),
                                                    'lon': array([-120.78559]),
                                                    'n_obs': array([141], dtype=int32),
                                                    'p_R': array([1.3853822e-32], dtype=float32),
                                                    'p_rho': array([4.62621e-39], dtype=float32),
                                                    'p_tau': array([nan], dtype=float32),
                                                    'rho': array([0.8418981], dtype=float32),
                                                    'tau': array([nan], dtype=float32)}}
    {(('ASCAT', 'sm'), ('ISMN', 'soil moisture')): {'BIAS': array([-0.6330102], dtype=float32),
                                                    'R': array([0.7807141], dtype=float32),
                                                    'RMSD': array([14.577002], dtype=float32),
                                                    'gpi': array([1], dtype=int32),
                                                    'lat': array([38.43003]),
                                                    'lon': array([-120.9675]),
                                                    'n_obs': array([482], dtype=int32),
                                                    'p_R': array([0.], dtype=float32),
                                                    'p_rho': array([0.], dtype=float32),
                                                    'p_tau': array([nan], dtype=float32),
                                                    'rho': array([0.6935607], dtype=float32),
                                                    'tau': array([nan], dtype=float32)}}
    {(('ASCAT', 'sm'), ('ISMN', 'soil moisture')): {'BIAS': array([-0.21823417], dtype=float32),
                                                    'R': array([0.80635566], dtype=float32),
                                                    'RMSD': array([12.903898], dtype=float32),
                                                    'gpi': array([2], dtype=int32),
                                                    'lat': array([38.17353]),
                                                    'lon': array([-120.80639]),
                                                    'n_obs': array([251], dtype=int32),
                                                    'p_R': array([0.], dtype=float32),
                                                    'p_rho': array([4.e-45], dtype=float32),
                                                    'p_tau': array([nan], dtype=float32),
                                                    'rho': array([0.74206454], dtype=float32),
                                                    'tau': array([nan], dtype=float32)}}
    {(('ASCAT', 'sm'), ('ISMN', 'soil moisture')): {'BIAS': array([-0.04437888], dtype=float32),
                                                    'R': array([0.6058206], dtype=float32),
                                                    'RMSD': array([17.388393], dtype=float32),
                                                    'gpi': array([3], dtype=int32),
                                                    'lat': array([34.783]),
                                                    'lon': array([-86.55]),
                                                    'n_obs': array([1652], dtype=int32),
                                                    'p_R': array([0.], dtype=float32),
                                                    'p_rho': array([0.], dtype=float32),
                                                    'p_tau': array([nan], dtype=float32),
                                                    'rho': array([0.62204134], dtype=float32),
                                                    'tau': array([nan], dtype=float32)}}
    {(('ASCAT', 'sm'), ('ISMN', 'soil moisture')): {'BIAS': array([0.2600247], dtype=float32),
                                                    'R': array([0.53643185], dtype=float32),
                                                    'RMSD': array([21.196829], dtype=float32),
                                                    'gpi': array([4], dtype=int32),
                                                    'lat': array([37.133]),
                                                    'lon': array([-97.083]),
                                                    'n_obs': array([1887], dtype=int32),
                                                    'p_R': array([0.], dtype=float32),
                                                    'p_rho': array([0.], dtype=float32),
                                                    'p_tau': array([nan], dtype=float32),
                                                    'rho': array([0.53143877], dtype=float32),
                                                    'tau': array([nan], dtype=float32)}}
    {(('ASCAT', 'sm'), ('ISMN', 'soil moisture')): {'BIAS': array([-0.1422875], dtype=float32),
                                                    'R': array([0.5070379], dtype=float32),
                                                    'RMSD': array([14.24668], dtype=float32),
                                                    'gpi': array([5], dtype=int32),
                                                    'lat': array([34.25]),
                                                    'lon': array([-105.417]),
                                                    'n_obs': array([1927], dtype=int32),
                                                    'p_R': array([0.], dtype=float32),
                                                    'p_rho': array([3.33e-42], dtype=float32),
                                                    'p_tau': array([nan], dtype=float32),
                                                    'rho': array([0.3029974], dtype=float32),
                                                    'tau': array([nan], dtype=float32)}}
    {(('ASCAT', 'sm'), ('ISMN', 'soil moisture')): {'BIAS': array([0.237454], dtype=float32),
                                                    'R': array([0.4996146], dtype=float32),
                                                    'RMSD': array([11.583476], dtype=float32),
                                                    'gpi': array([6], dtype=int32),
                                                    'lat': array([33.8833]),
                                                    'lon': array([102.1333]),
                                                    'n_obs': array([357], dtype=int32),
                                                    'p_R': array([6.127213e-24], dtype=float32),
                                                    'p_rho': array([2.471651e-28], dtype=float32),
                                                    'p_tau': array([nan], dtype=float32),
                                                    'rho': array([0.53934574], dtype=float32),
                                                    'tau': array([nan], dtype=float32)}}
    {(('ASCAT', 'sm'), ('ISMN', 'soil moisture')): {'BIAS': array([-0.04330891], dtype=float32),
                                                    'R': array([0.7128256], dtype=float32),
                                                    'RMSD': array([7.729667], dtype=float32),
                                                    'gpi': array([7], dtype=int32),
                                                    'lat': array([33.6666]),
                                                    'lon': array([102.1333]),
                                                    'n_obs': array([384], dtype=int32),
                                                    'p_R': array([0.], dtype=float32),
                                                    'p_rho': array([0.], dtype=float32),
                                                    'p_tau': array([nan], dtype=float32),
                                                    'rho': array([0.7002289], dtype=float32),
                                                    'tau': array([nan], dtype=float32)}}


The validation is then performed by looping over all the defined jobs
and storing the results. You can see that the results are a dictionary
where the key is a tuple defining the exact combination of datasets and
columns that were used for the calculation of the metrics. The metrics
itself are a dictionary of ``metric-name:  numpy.ndarray`` which also
include information about the gpi, lon and lat. Since all the
information contained in the job is given to the metric calculator they
can be stored in the results.

Storing of the results to disk is at the moment supported by the
``netcdf_results_manager`` which creates a netCDF file for each dataset
combination and stores each metric as a variable. We can inspect the
stored netCDF file which is named after the dictionary key:

.. code:: ipython2

    import netCDF4
    results_fname = os.path.join(save_path, 'ASCAT.sm_with_ISMN.soil moisture.nc')
    
    with netCDF4.Dataset(results_fname) as ds:
        for var in ds.variables:
            print var, ds.variables[var][:]


.. parsed-literal::

    n_obs [141 482 251 1652 1887 1927 357 384 141 482 251 1652 1887 1927 357 384 141
     482 141 482 251 1652 1887 1927 357 384 141 482 251 1652 1887 1927 357 384
     141 482 251 1652 1887 1927 357 384]
    tau [nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
     nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
     nan nan nan nan nan nan]
    gpi [0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2
     3 4 5 6 7]
    RMSD [13.06224250793457 14.577001571655273 12.903898239135742 17.38839340209961
     21.196828842163086 14.24668025970459 11.583476066589355
     7.7296671867370605 13.06224250793457 14.577001571655273
     12.903898239135742 17.38839340209961 21.196828842163086 14.24668025970459
     11.583476066589355 7.7296671867370605 13.06224250793457
     14.577001571655273 13.06224250793457 14.577001571655273
     12.903898239135742 17.38839340209961 21.196828842163086 14.24668025970459
     11.583476066589355 7.7296671867370605 13.06224250793457
     14.577001571655273 12.903898239135742 17.38839340209961
     21.196828842163086 14.24668025970459 11.583476066589355
     7.7296671867370605 13.06224250793457 14.577001571655273
     12.903898239135742 17.38839340209961 21.196828842163086 14.24668025970459
     11.583476066589355 7.7296671867370605]
    lon [-120.78559 -120.9675 -120.80639 -86.55 -97.083 -105.417 102.1333 102.1333
     -120.78559 -120.9675 -120.80639 -86.55 -97.083 -105.417 102.1333 102.1333
     -120.78559 -120.9675 -120.78559 -120.9675 -120.80639 -86.55 -97.083
     -105.417 102.1333 102.1333 -120.78559 -120.9675 -120.80639 -86.55 -97.083
     -105.417 102.1333 102.1333 -120.78559 -120.9675 -120.80639 -86.55 -97.083
     -105.417 102.1333 102.1333]
    p_tau [nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
     nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
     nan nan nan nan nan nan]
    BIAS [-1.9682410955429077 -0.63301020860672 -0.21823416650295258
     -0.04437888041138649 0.26002469658851624 -0.1422874927520752
     0.23745399713516235 -0.043308909982442856 -1.9682410955429077
     -0.63301020860672 -0.21823416650295258 -0.04437888041138649
     0.26002469658851624 -0.1422874927520752 0.23745399713516235
     -0.043308909982442856 -1.9682410955429077 -0.63301020860672
     -1.9682410955429077 -0.63301020860672 -0.21823416650295258
     -0.04437888041138649 0.26002469658851624 -0.1422874927520752
     0.23745399713516235 -0.043308909982442856 -1.9682410955429077
     -0.63301020860672 -0.21823416650295258 -0.04437888041138649
     0.26002469658851624 -0.1422874927520752 0.23745399713516235
     -0.043308909982442856 -1.9682410955429077 -0.63301020860672
     -0.21823416650295258 -0.04437888041138649 0.26002469658851624
     -0.1422874927520752 0.23745399713516235 -0.043308909982442856]
    p_rho [4.6262103163618786e-39 0.0 4.203895392974451e-45 0.0 0.0
     3.3294851512357654e-42 2.471651101555352e-28 0.0 4.6262103163618786e-39
     0.0 4.203895392974451e-45 0.0 0.0 3.3294851512357654e-42
     2.471651101555352e-28 0.0 4.6262103163618786e-39 0.0
     4.6262103163618786e-39 0.0 4.203895392974451e-45 0.0 0.0
     3.3294851512357654e-42 2.471651101555352e-28 0.0 4.6262103163618786e-39
     0.0 4.203895392974451e-45 0.0 0.0 3.3294851512357654e-42
     2.471651101555352e-28 0.0 4.6262103163618786e-39 0.0
     4.203895392974451e-45 0.0 0.0 3.3294851512357654e-42
     2.471651101555352e-28 0.0]
    rho [0.8418980836868286 0.6935607194900513 0.7420645356178284
     0.6220413446426392 0.5314387679100037 0.3029974102973938
     0.5393457412719727 0.7002289295196533 0.8418980836868286
     0.6935607194900513 0.7420645356178284 0.6220413446426392
     0.5314387679100037 0.3029974102973938 0.5393457412719727
     0.7002289295196533 0.8418980836868286 0.6935607194900513
     0.8418980836868286 0.6935607194900513 0.7420645356178284
     0.6220413446426392 0.5314387679100037 0.3029974102973938
     0.5393457412719727 0.7002289295196533 0.8418980836868286
     0.6935607194900513 0.7420645356178284 0.6220413446426392
     0.5314387679100037 0.3029974102973938 0.5393457412719727
     0.7002289295196533 0.8418980836868286 0.6935607194900513
     0.7420645356178284 0.6220413446426392 0.5314387679100037
     0.3029974102973938 0.5393457412719727 0.7002289295196533]
    lat [38.14956 38.43003 38.17353 34.783 37.133 34.25 33.8833 33.6666 38.14956
     38.43003 38.17353 34.783 37.133 34.25 33.8833 33.6666 38.14956 38.43003
     38.14956 38.43003 38.17353 34.783 37.133 34.25 33.8833 33.6666 38.14956
     38.43003 38.17353 34.783 37.133 34.25 33.8833 33.6666 38.14956 38.43003
     38.17353 34.783 37.133 34.25 33.8833 33.6666]
    R [0.7996008396148682 0.7807140946388245 0.8063556551933289
     0.6058205962181091 0.5364318490028381 0.507037878036499
     0.4996145963668823 0.71282559633255 0.7996008396148682 0.7807140946388245
     0.8063556551933289 0.6058205962181091 0.5364318490028381
     0.507037878036499 0.4996145963668823 0.71282559633255 0.7996008396148682
     0.7807140946388245 0.7996008396148682 0.7807140946388245
     0.8063556551933289 0.6058205962181091 0.5364318490028381
     0.507037878036499 0.4996145963668823 0.71282559633255 0.7996008396148682
     0.7807140946388245 0.8063556551933289 0.6058205962181091
     0.5364318490028381 0.507037878036499 0.4996145963668823 0.71282559633255
     0.7996008396148682 0.7807140946388245 0.8063556551933289
     0.6058205962181091 0.5364318490028381 0.507037878036499
     0.4996145963668823 0.71282559633255]
    p_R [1.3853822467078656e-32 0.0 0.0 0.0 0.0 0.0 6.12721281290096e-24 0.0
     1.3853822467078656e-32 0.0 0.0 0.0 0.0 0.0 6.12721281290096e-24 0.0
     1.3853822467078656e-32 0.0 1.3853822467078656e-32 0.0 0.0 0.0 0.0 0.0
     6.12721281290096e-24 0.0 1.3853822467078656e-32 0.0 0.0 0.0 0.0 0.0
     6.12721281290096e-24 0.0 1.3853822467078656e-32 0.0 0.0 0.0 0.0 0.0
     6.12721281290096e-24 0.0]


Parallel processing
-------------------

The same code can be executed in parallel by defining the following
``start_processing`` function.

.. code:: ipython2

    def start_processing(job):
        try:
            return process.calc(*job)
        except RuntimeError:
            return process.calc(*job)

``pytesmo.validation_framework.start_validation`` can then be used to
run your validation in parallel. Your setup code can look like this
Ipython notebook without the loop over the jobs. Otherwise the
validation would be done twice. Save it into a ``.py`` file e.g.
``my_validation.py``.

After `starting the ipyparallel
cluster <http://ipyparallel.readthedocs.org/en/latest/process.html>`__
you can then execute the following code:

.. code:: python

    from pytesmo.validation_framework import start_validation

    # Note that before starting the validation you must start a controller
    # and engines, for example by using: ipcluster start -n 4
    # This command will launch a controller and 4 engines on the local machine.
    # Also, do not forget to change the setup_code path to your current setup.

    setup_code = "my_validation.py"
    start_validation(setup_code)

Masking datasets
----------------

Masking datasets are datasets that return a pandas DataFrame with
boolean values. ``True`` means that the observation should be masked,
``False`` means it should be kept. All masking datasets are temporally
matched in pairs to the temporal reference dataset. Only observations
for which all masking datasets have a value of ``False`` are kept for
further validation.

The masking datasets have the same format as the dataset dictionary and
can be specified in the Validation class with the ``masking_datasets``
keyword.

Masking adapter
~~~~~~~~~~~~~~~

To easily transform an existing dataset into a masking dataset
``pytesmo`` offers a adapter class that calls the ``read_ts`` method of
an existing dataset and creates a masking dataset based on an operator,
a given threshold, and (optionally) a column name.

.. code:: ipython2

    from pytesmo.validation_framework.adapters import MaskingAdapter
    
    ds_mask = MaskingAdapter(ismn_reader, '<', 0.2, 'soil moisture')
    print ds_mask.read_ts(ids[0]).head()


.. parsed-literal::

                         soil moisture
    date_time                         
    2012-12-14 19:00:00          False
    2012-12-14 20:00:00          False
    2012-12-14 21:00:00          False
    2012-12-14 22:00:00          False
    2012-12-14 23:00:00          False


Self-masking adapter
~~~~~~~~~~~~~~~~~~~~

``pytesmo`` also has a class that masks a dataset "on-the-fly", based on
one of the columns it contains and an operator and a threshold. In
contrast to the masking adapter mentioned above, the output of the
self-masking adapter is the masked data, not the the mask. The
self-masking adapter wraps a data reader, which must have a ``read_ts``
or ``read`` method. Calling its ``read_ts``/``read`` method will return
the masked data - more precisely a DataFrame with only rows where the
masking condition is true.

.. code:: ipython2

    from pytesmo.validation_framework.adapters import SelfMaskingAdapter
    
    ds_mask = SelfMaskingAdapter(ismn_reader, '<', 0.2, 'soil moisture')
    print ds_mask.read_ts(ids[0]).head()


.. parsed-literal::

                         soil moisture soil moisture_flag  soil moisture_orig_flag
    date_time                                                                     
    2013-08-21 22:00:00         0.1682                  U                        0
    2013-08-21 23:00:00         0.1665                  U                        0
    2013-08-22 00:00:00         0.1682                  U                        0
    2013-08-22 01:00:00         0.1615                  U                        0
    2013-08-22 02:00:00         0.1631                  U                        0

