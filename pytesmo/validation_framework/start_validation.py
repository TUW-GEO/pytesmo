from IPython import parallel
try:
    from itertools import izip as zip
except ImportError:
    # python 3
    pass
from datetime import datetime

from pytesmo.validation_framework.results_manager import netcdf_results_manager


def func(job):
    """
    Function which calls the start_processing method implemented in setup_code.
    """
    return start_processing(job)


def start_validation(setup_code):
    """
    Perform the validation with IPython parallel processing.

    Parameters
    ----------
    setup_code : string
        Path to .py file containing the setup for the validation.
    """
    c = parallel.Client()
    dv = c[:]
    lview = c.load_balanced_view()

    dv.run(setup_code, block=True)

    jobs = None
    try:
        jobs = dv['jobs'][0]
    except parallel.CompositeError:
        print("Variable 'jobs' is not defined!")

    save_path = None
    try:
        save_path = dv['save_path'][0]
    except parallel.CompositeError:
        print("Variable 'save_path' is not defined!")

    to_write = len(jobs)
    if (jobs is not None) and (save_path is not None):
        with lview.temp_flags(retries=2):
            amr = lview.map_async(func, jobs)
            results = zip(amr, jobs)
            for result, job in results:
                netcdf_results_manager(result, save_path)
                to_write -= 1
                print('job = ' + str(job), 'remaining jobs = ' + str(to_write))

    c[:].clear()


if __name__ == '__main__':

    start = datetime.now()
    print('Start Validation')

    # Note that before starting the validation you must start a controller
    # and engines, for example by using: ipcluster start -n 4
    # This command will launch a controller and 4 engines on the local machine.
    # Also, do not forget to change the setup_code path to your current setup.
    setup_code = "/media/sf_H/swdvlp/github/pytesmo/examples/setup_validation_ASCAT_ISMN.py"
    start_validation(setup_code)

    print('Elapsed time:', datetime.now() - start)
