
# coding: utf-8

# # Example soil moisture validation: ASCAT - ISMN

# This example shows how to setup the pytesmo validation framework to perform the validation either normal or using the parallel processing tool from ipython. 

# In[ ]:

import os

import pytesmo.validation_framework.temporal_matchers as temporal_matchers
import pytesmo.validation_framework.metric_calculators as metrics_calculators

from datetime import datetime

from pytesmo.io.sat.ascat import AscatH25_SSM
from pytesmo.io.ismn.interface import ISMN_Interface
from pytesmo.validation_framework.validation import Validation

from examples.data_preparation_ASCAT_ISMN import DataPreparation


# Initialize ASCAT reader

# In[ ]:

ascat_data_folder = os.path.join('/media/sf_R', 'Datapool_processed', 'WARP', 'WARP5.5',
                                 'IRMA1_WARP5.5_P2', 'R1', '080_ssm', 'netcdf')
ascat_grid_folder = os.path.join('/media/sf_R', 'Datapool_processed', 'WARP',
                                 'ancillary', 'warp5_grid')

ascat_reader = AscatH25_SSM(ascat_data_folder, ascat_grid_folder)
ascat_reader.read_bulk = True
ascat_reader._load_grid_info()


# Initialize ISMN reader

# In[ ]:

ismn_data_folder = os.path.join('/media/sf_D', 'ISMN', 'data')
ismn_reader = ISMN_Interface(ismn_data_folder)


# Create the variable ***jobs*** which is a list containing either cell numbers (for a cell based process) or grid point index information tuple(gpi, longitude, latitude). For ISMN *gpi* is replaced by *idx* which is an index used to read time series of variables such as soil moisture. **DO NOT CHANGE** the name ***jobs*** because it will be searched during the parallel processing!

# In[ ]:

jobs = []

ids = ismn_reader.get_dataset_ids(variable='soil moisture', min_depth=0, max_depth=0.1)
for idx in ids:
    metadata = ismn_reader.metadata[idx]
    jobs.append((idx, metadata['longitude'], metadata['latitude']))


# Create the variable ***save_path*** which is a string representing the path where the results will be saved. **DO NOT CHANGE** the name ***save_path*** because it will be searched during the parallel processing!

# In[ ]:

save_path = os.path.join('/media/sf_D', 'validation_framework', 'test_ASCAT_ISMN')


# Create the validation object.

# In[ ]:

datasets = {'ISMN': {'class': ismn_reader, 'columns': ['soil moisture'],
                     'type': 'reference', 'args': [], 'kwargs': {}},
            'ASCAT': {'class': ascat_reader, 'columns': ['sm'], 'type': 'other',
                      'args': [], 'kwargs': {}, 'grids_compatible': False,
                      'use_lut': False, 'lut_max_dist': 30000}
            }

period = [datetime(2007, 1, 1), datetime(2014, 12, 31)]

process = Validation(datasets=datasets, data_prep=DataPreparation(),
                     temporal_matcher=temporal_matchers.BasicTemporalMatching(window=1/24.0, reverse=True),
                     scaling='lin_cdf_match', scale_to_other=True,
                     metrics_calculator=metrics_calculators.BasicMetrics(),
                     period=period, cell_based_jobs=False)


# * If you decide to use the **ipython parallel processing** to perform the validation please **ADD** the ***start_processing*** function to your code. Then move to pytesmo.validation_framework.start_validation, change the path to your setup code and start the validation.

# In[ ]:

def start_processing(job):
    try:
        return process.calc(job)
    except RuntimeError:
        return process.calc(job)


# * If you chose to perform the **validation normally** then please **ADD** the uncommented ***main*** method to your code.

# In[ ]:

# if __name__ == '__main__':
# 
#     from pytesmo.validation_framework.results_manager import netcdf_results_manager
# 
#     for job in jobs:
#         results = process.calc(job)
#         netcdf_results_manager(results, save_path)

