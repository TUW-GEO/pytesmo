import sys
sys.path.append('/media/sf_H/swdvlp/GEO_Python_cvs')
sys.path.append('/media/sf_H/swdvlp/RS.Developments/rs-data-readers')
sys.path.append('/media/sf_H/swdvlp/RS.Developments/pygrids')
sys.path.append('/media/sf_H/swdvlp/github/pygeogrids')
sys.path.append('/media/sf_H/swdvlp/RS.Developments/pynetcf')

import rsdata.GLDAS_NOAH.interface as GLDAS

import general.root_path as root
import pytesmo.validation_framework.metric_calculators as metrics_calculators
import pytesmo.validation_framework.temporal_matchers as temporal_matchers
from pytesmo.validation_framework.validation import Validation
import pytesmo.validation_framework.result_managers as result_managers

from datetime import datetime
import os
import general.io.compr_pickle as pickle

import pytesmo.io.sat.ascat as ASCAT

ascat_data_path = os.path.join(root.d, 'validation_framework', 'ASCAT_data')
ascat_grid_path = os.path.join(root.r, 'Datapool_processed', 'WARP',
                               'ancillary', 'warp5_grid')

ascat = ASCAT.AscatH25_SSM(path=ascat_data_path, grid_path=ascat_grid_path)
ascat._load_grid_info()

datasets = {'ASCAT': ascat,
            'GLDAS': GLDAS.GLDAS025v1_nc(parameter='086_L1')}

reference_column = ['sm']
other_column = ['086_L1']

period = [datetime(2007, 1, 1), datetime(2007, 1, 31)]

save_path = os.path.join(root.d, 'validation_framework', 'results')

process = Validation(datasets=datasets, reference_name='ASCAT',
                     other_name='GLDAS', reference_column=reference_column,
                     other_column=other_column, reference_args=[],
                     other_args=[],
                     reference_kwargs={'mask_frozen_prob': 80,
                                       'mask_snow_prob': 80,
                                       'mask_ssf': True}, other_kwargs={},
                     grids_compatible=False, data_prep=None,
                     temporal_matcher=temporal_matchers.BasicTemporalMatching(),
                     scaling='lin_cdf_match', scale_to_other=False,
                     metrics_calculator=metrics_calculators.BasicMetrics(),
                     result_names=reference_column, use_lut=False,
                     lut_max_dist=30000, period=period, cell_based_jobs=True,
                     save_path=save_path, result_man=None)

jobs = [1246, 2045]


def get_jobs():
    return jobs


def start_processing(job):
    return process.calc(job)


if __name__ == '__main__':

    result = process.calc(jobs[0])
    print 'ok'
