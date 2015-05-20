from IPython import parallel
from itertools import izip
from datetime import datetime

from pytesmo.validation_framework.setup_validation import get_jobs


def func(job):
    return start_processing(job)


def start_validation():

    c = parallel.Client()
    dv = c[:]
    lview = c.load_balanced_view()

    dv.run("setup_validation.py", block=True)

    jobs = get_jobs()
    with lview.temp_flags(retries=2):
        amr = lview.map_async(func, jobs)
        results = izip(amr, jobs)
        for result, job in results:
            print 'ok'

    c[:].clear()


if __name__ == '__main__':

    start = datetime.now()
    print 'Start Validation'
    start_validation()
    print 'Elapsed time:', datetime.now() - start
