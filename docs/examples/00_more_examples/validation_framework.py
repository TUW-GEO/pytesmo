
# coding: utf-8

# # The pytesmo validation framework
# 
# The pytesmo validation framework takes care of iterating over datasets, spatial and temporal matching as well as
# scaling. It uses metric calculators to then calculate metrics that are returned to the user. There are several
# metrics calculators included in pytesmo but new ones can be added simply.
# 
# ## Overview
# 
# How does the validation framework work? It makes these assumptions about the used datasets:
# 
# - The dataset readers that are used have a `read_ts` method that can be called either by a grid point index (gpi)
#   which can be any indicator that identifies a certain grid point or by using longitude and latitude. This means that
#   both call signatures `read_ts(gpi)` and `read_ts(lon, lat)` must be valid. Please check the
#   [pygeobase](https://github.com/TUW-GEO/pygeobase) documentation for more details on how a fully compatible dataset
#   class should look. But a simple `read_ts` method should do for the validation framework. This assumption can be
#   relaxed by using the `read_ts_names` keyword in the pytesmo.validation_framework.data_manager.DataManager class.
# - The `read_ts` method returns a pandas.DataFrame time series.
# - Ideally the datasets classes also have a `grid` attribute that is a
#   [pygeogrids](http://pygeogrids.readthedocs.org/en/latest/) grid. This makes the calculation of lookup tables easily
#   possible and the nearest neighbor search faster.
# 
# Fortunately these assumptions are true about the dataset readers included in pytesmo. 
# 
# It also makes a few assumptions about how to perform a validation. For a comparison study it is often necessary to
# choose a spatial reference grid, a temporal reference and a scaling or data space reference.
# 
# ### Spatial reference
# The spatial reference is the one to which all the other datasets are matched spatially. Often through nearest
# neighbor search. The validation framework uses grid points of the dataset specified as the spatial reference to
# spatially match all the other datasets with nearest neighbor search. Other, more sophisticated spatial matching
# algorithms are not implemented at the moment. If you need a more complex spatial matching then a preprocessing of
# the data is the only option at the moment.
# 
# ### Temporal reference
# The temporal reference is the dataset to which the other dataset are temporally matched. That means that the
# nearest observation to the reference timestamps in a certain time window is chosen for each comparison dataset.
# This is by default done by the temporal matching module included in pytesmo. How many datasets should be matched to
# the reference dataset at once can be configured, we will cover how to do this later.
# 
# ### Data space reference
# It is often necessary to bring all the datasets into a common data space by using. Scaling is often used for that
# and pytesmo offers a choice of several scaling algorithms (e.g. CDF matching, min-max scaling, mean-std scaling,
# triple collocation based scaling). The data space reference can also be chosen independently from the other two
# references.
# 
# ## Data Flow
# 
# After it is initialized, the validation framework works through the following steps:
# 
# 1. Read all the datasets for a certain job (gpi, lon, lat)
# 2. Read all the masking dataset if any
# 3. Mask the temporal reference dataset using the masking data
# 4. Temporally match all the chosen combinations of temporal reference and other datasets
# 5. Turn the temporally matched time series over to the metric calculators
# 6. Get the calculated metrics from the metric calculators
# 7. Put all the metrics into a dictionary by dataset combination and return them.
# 
# ## Masking datasets
# Masking datasets can be used if the datasets that are compared do not contain the necessary information to mask
#  them. For example we might want to use modelled soil temperature data to mask our soil moisture observations
# before comparing them. To be able to do that we just need a Dataset that returns a pandas.DataFrame with one column
#  of boolean data type. Everywhere where the masking dataset is `True` the data will be masked.
# 
# Let's look at a first example.
# 
# ## Example soil moisture validation: ASCAT - ISMN
# 
# This example shows how to setup the pytesmo validation framework to perform a comparison between ASCAT and ISMN data. 
# 

# In[1]:


import os

import pytesmo.validation_framework.metric_calculators as metrics_calculators

from datetime import datetime

from ascat.read_native.cdr import AscatSsmCdr
from ismn.interface import ISMN_Interface
from pytesmo.validation_framework.validation import Validation
from pytesmo.validation_framework.results_manager import netcdf_results_manager

# You need the test data from https://github.com/TUW-GEO/pytesmo-test-data for this example

# In[2]:


testdata_folder = '/pytesmo/testdata'
output_folder = '/pytesmo/code/examples/output'


# First we initialize the data readers that we want to use. In this case the ASCAT soil moisture time series and in
# situ data from the ISMN.
# 
# Initialize ASCAT reader

# In[3]:


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


# Initialize ISMN reader

# In[4]:


ismn_data_folder = os.path.join(testdata_folder,
                                 'ismn/multinetwork/header_values')

ismn_reader = ISMN_Interface(ismn_data_folder)


# The validation is run based on jobs. A job consists of at least three lists or numpy arrays specifing the grid
# point index, its latitude and longitude. In the case of the ISMN we can use the `dataset_ids` that identify every
# time series in the downloaded ISMN data as our grid point index. We can then get longitude and latitude from the
# metadata of the dataset.
# 
# **DO NOT CHANGE** the name ***jobs*** because it will be searched during the parallel processing!

# In[5]:


jobs = []

ids = ismn_reader.get_dataset_ids(variable='soil moisture', min_depth=0, max_depth=0.1)
for idx in ids:
    metadata = ismn_reader.metadata[idx]
    jobs.append((idx, metadata['longitude'], metadata['latitude']))

print("Jobs (gpi, lon, lat):")
print(jobs)


# For this small test dataset it is only one job
# 
# It is important here that the ISMN reader has a read_ts function that works by just using the `dataset_id`. In this
#  way the validation framework can go through the jobs and read the correct time series.

# In[6]:


data = ismn_reader.read_ts(ids[0])
print('ISMN data example:')
print(data.head())


# ## Initialize the Validation class
# 
# The Validation class is the heart of the validation framework. It contains the information about which datasets to
# read using which arguments or keywords and if they are spatially compatible. It also contains the settings about
# which metric calculators to use and how to perform the scaling into the reference data space. It is initialized in
# the following way:

# In[7]:

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

# The datasets dictionary contains all the information about the datasets to read. The `class` is the dataset class
# to use which we have already initialized. The `columns` key describes which columns of the dataset interest us for
# validation. This a mandatory field telling the framework which other columns to ignore. In this case the columns
# `soil moisture_flag` and `soil moisture_orig_flag` will be ignored by the ISMN reader. We can also specify
# additional keywords that should be given to the `read_ts` method of the dataset reader. In this case we want the
# ASCAT reader to mask the ASCAT soil moisture using the included frozen and snow probabilities as well as the SSF.
# There are also other keys that can be used here. Please see the documentation for explanations.

# In[8]:


period = [datetime(2007, 1, 1), datetime(2014, 12, 31)]
basic_metrics = metrics_calculators.BasicMetrics(other_name='k1')

process = Validation(
    datasets, 'ISMN',
    temporal_ref='ASCAT',
    scaling='lin_cdf_match',
    scaling_ref='ASCAT',   
    metrics_calculators={(2, 2): basic_metrics.calc_metrics},
    period=period)


# During the initialization of the Validation class we can also tell it other things that it needs to know. In this
# case it uses the datasets we have specified earlier. The spatial reference is the `'ISMN'` dataset which is the
# second argument. The 'metrics_calculators' argument looks a little bit strange so let's look at it in more detail.
# 
# It is a dictionary with a tuple as the key and a function as the value. The key tuple `(n, k)` has the following
# meaning: `n` datasets are temporally matched together and then given in sets of `k` columns to the metric
# calculator. The metric calculator then gets a DataFrame with the columns ['ref', 'k1', 'k2' ...] and so on
# depending on the value of k. The value of `(2, 2)` makes sense here since we only have two datasets and all our
# metrics also take two inputs.
# 
# This can be used in more complex scenarios to e.g. have three input datasets that are all temporally matched
# together and then combinations of two input datasets are given to one metric calculator while all three datasets
# are given to another metric calculator. This could look like this:
# 
# ```python
# { (3 ,2): metric_calc,
#   (3, 3): triple_collocation}
# ```
# 
# Create the variable ***save_path*** which is a string representing the path where the results will be saved.
# **DO NOT CHANGE** the name ***save_path*** because it will be searched during the parallel processing!
# In[9]:

save_path = output_folder

import pprint
for job in jobs:
    
    results = process.calc(*job)
    pprint.pprint(results)
    netcdf_results_manager(results, save_path)


# The validation is then performed by looping over all the defined jobs and storing the results.
# You can see that the results are a dictionary where the key is a tuple defining the exact combination of datasets
# and columns that were used for the calculation of the metrics. The metrics itself are a dictionary of `metric-name:
#  numpy.ndarray` which also include information about the gpi, lon and lat. Since all the information contained in
# the job is given to the metric calculator they can be stored in the results.
# 
# Storing of the results to disk is at the moment supported by the `netcdf_results_manager` which creates a netCDF
# file for each dataset combination and stores each metric as a variable. We can inspect the stored netCDF file which
#  is named after the dictionary key:

# In[10]:


import netCDF4
results_fname = os.path.join(save_path, 'ASCAT.sm_with_ISMN.soil moisture.nc')

with netCDF4.Dataset(results_fname) as ds:
    for var in ds.variables:
        print(var, ds.variables[var][:])


# ## Parallel processing
# 
# The same code can be executed in parallel by defining the following `start_processing` function.

# In[11]:


def start_processing(job):
    try:
        return process.calc(*job)
    except RuntimeError:
        return process.calc(*job)


# `pytesmo.validation_framework.start_validation` can then be used to run your validation in parallel.
# Your setup code can look like this Ipython notebook without the loop over the jobs. Otherwise the validation would
# be done twice. Save it into a `.py` file e.g. `my_validation.py`.
# 
# After [starting the ipyparallel cluster](http://ipyparallel.readthedocs.org/en/latest/process.html) you can then
# execute the following code:
# 
# ```python
# from pytesmo.validation_framework import start_validation
# 
# # Note that before starting the validation you must start a controller
# # and engines, for example by using: ipcluster start -n 4
# # This command will launch a controller and 4 engines on the local machine.
# # Also, do not forget to change the setup_code path to your current setup.
# 
# setup_code = "my_validation.py"
# start_validation(setup_code)
# ```
# 
# ## Masking datasets
# 
# Masking datasets are datasets that return a pandas DataFrame with boolean values. `True` means that the observation
#  should be masked, `False` means it should be kept. All masking datasets are temporally matched in pairs to the
# temporal reference dataset. Only observations for which all masking datasets have a value of `False` are kept for
# further validation.
# 
# The masking datasets have the same format as the dataset dictionary and can be specified in the Validation class
# with the `masking_datasets` keyword.
# 
# ### Masking adapter
# 
# To easily transform an existing dataset into a masking dataset `pytesmo` offers a adapter class that calls the
# `read_ts` method of an existing dataset and creates a masking dataset based on an operator, a given threshold, and (optionally) a column name.

# In[12]:


from pytesmo.validation_framework.adapters import MaskingAdapter

ds_mask = MaskingAdapter(ismn_reader, '<', 0.2, 'soil moisture')
print(ds_mask.read_ts(ids[0]).head())


# ### Self-masking adapter
# `pytesmo` also has a class that masks a dataset "on-the-fly", based on one of the columns it contains and an operator and a threshold. In contrast to the masking adapter mentioned above, the output of the self-masking adapter is the masked data, not the the mask. The self-masking adapter wraps a data reader, which must have a `read_ts` or `read` method. Calling its `read_ts`/`read` method will return the masked data - more precisely a DataFrame with only rows where the masking condition is true.

# In[13]:


from pytesmo.validation_framework.adapters import SelfMaskingAdapter

ds_mask = SelfMaskingAdapter(ismn_reader, '<', 0.2, 'soil moisture')
print(ds_mask.read_ts(ids[0]).head())

