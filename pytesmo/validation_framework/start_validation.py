from IPython import parallel
from itertools import izip
from datetime import datetime

from pytesmo.validation_framework.results_manager import netcdf_results_manager


def func(job):
    return start_processing(job)


def start_validation():

    c = parallel.Client()
    dv = c[:]
    lview = c.load_balanced_view()

    dv.run("/media/sf_H/swdvlp/aplocon/code/Validation/ASCAT_soil_moisture/setup_validation.py", block=True)

    jobs = None
    try:
        jobs = dv['jobs'][0]
    except parallel.CompositeError:
        print "Variable 'jobs' is not defined!"

    save_path = None
    try:
        save_path = dv['save_path'][0]
    except parallel.CompositeError:
        print "Variable 'save_path' is not defined!"

    if (jobs is not None) and (save_path is not None):
        with lview.temp_flags(retries=2):
            amr = lview.map_async(func, jobs)
            results = izip(amr, jobs)
            for result, job in results:
                netcdf_results_manager(result, save_path)
                print job

    c[:].clear()


if __name__ == '__main__':

    start = datetime.now()
    print 'Start Validation'
    start_validation()
    print 'Elapsed time:', datetime.now() - start
