pytesmo - a Python Toolbox for the Evaluation of Soil Moisture Observations
***************************************************************************

.. image:: https://github.com/TUW-GEO/pytesmo/workflows/tests/badge.svg
   :target: https://github.com/TUW-GEO/pytesmo/actions?query=tests

.. image:: https://coveralls.io/repos/TUW-GEO/pytesmo/badge.png?branch=master
  :target: https://coveralls.io/r/TUW-GEO/pytesmo?branch=master
  
.. image:: https://badge.fury.io/py/pytesmo.svg
    :target: https://badge.fury.io/py/pytesmo

.. image:: https://readthedocs.org/projects/pytesmo/badge/?version=latest
    :target: https://pytesmo.readthedocs.io/en/latest/?badge=latest

pytesmo, the Python Toolbox for the Evaluation of Soil Moisture Observations, is
a package/python toolbox which aims it is to provide a library that can be used
for the comparison and validation of geospatial time series datasets with a
(initial) focus on soil moisture.

Documentation & Software Citation
=================================

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.596422.svg
   :target: https://doi.org/10.5281/zenodo.596422

To see the latest `full documentation <https://pytesmo.readthedocs.io/en/latest/?badge=latest>`_
click on the docs badge at the top.

If you use the software in a publication then please cite it using the Zenodo DOI.
Be aware that this badge links to the latest package version.

Please select your specific version at https://doi.org/10.5281/zenodo.596422 to get the DOI of that version.
You should normally always use the DOI for the specific version of your record in citations.
This is to ensure that other researchers can access the exact research artefact you used for reproducibility.

You can find additional information regarding DOI versioning at http://help.zenodo.org/#versioning

Installation
============

This package should be installable through `pip
<https://pip.pypa.io/en/latest/installing.html>`__ which downloads the package
from the `python package repository Pypi <https://pypi.python.org/>`__.
However, pytesmo also needs some packages that depend on C or Fortran libraries (like ``netCDF4``).
They should be installed first with conda. See http://conda.pydata.org/docs/ on how to use it.
We recommend using either `Anaconda <https://www.anaconda.com/>`__ or
`Miniconda <https://conda.io/en/latest/miniconda.html>`__.

.. code-block:: bash

    $ conda install -c conda-forge numpy scipy pandas netCDF4 cython pyresample pyresample

Afterwards ``pytesmo`` can be installed via pip.

.. code-block:: bash

    $ pip install pytesmo


You can also install all needed (conda and pip) dependencies at once using the following
commands after cloning this repository.
This is recommended for developers of the package. Note that the git ``--recursive`` flag
will clone test-data, which is needed by some tests.

.. code::

    $ git clone https://github.com/TUW-GEO/pytesmo.git --recursive
    $ cd pytesmo
    $ conda create -n pytesmo python=3.6 # or any supported python version
    $ source activate pytesmo
    $ conda update -f environment.yml -n pytesmo
    $ python setup.py develop

.. note::

    If you are using windows and conda is missing a package then always check
    http://www.lfd.uci.edu/~gohlke/pythonlibs/ to see if there is already a
    precompiled .exe or .whl file for you to easily install.

Supported Products
==================

Soil moisture is observed using different methods and instruments, in this
version several satellite datasets as well as in situ and reanalysis data are supported
through independent and optional (reader) packages:

- `ERS & H-SAF ASCAT products <https://github.com/TUW-GEO/ascat/>`_
- `SMAP <https://github.com/TUW-GEO/smap_io/>`_
- `GLDAS Noah <https://github.com/TUW-GEO/gldas/>`_
- `ERA5 and ERA5-Land <https://github.com/TUW-GEO/ecmwf_models/>`_
- `SMOS <https://github.com/TUW-GEO/smos/>`_
- `C3S SM <https://github.com/TUW-GEO/c3s_sm/>`_
- `ESA CCI SM <https://github.com/TUW-GEO/esa_cci_sm/>`_
- `MERRA <https://github.com/TUW-GEO/merra/>`_
- `Data from the International Soil Moisture Network (ISMN) <https://github.com/TUW-GEO/ismn/>`_
    In case of the ISMN, two different formats are provided:
    An example of how to use the dataset in the pytesmo validation framework can be
    found in the "Examples" chapter.
    * Variables stored in separate files (CEOP formatted)

Contribute
==========

We are happy if you want to contribute. Please raise an issue explaining what
is missing or if you find a bug. We will also gladly accept pull requests
against our master branch for new features or bug fixes.

**Guidelines**

If you want to contribute please follow these steps:

- Fork the pytesmo repository to your account
- Clone the repository, make sure you use ``git clone --recursive`` to also get
  the test data repository.
- make a new feature branch from the pytesmo master branch
- Add your feature
- please include tests for your contributions in one of the test directories
  We use py.test so a simple function called test_my_feature is enough
- submit a pull request to our master branch

**Release Windows**

In order to make a working release for windows the ``whl`` files for windows
from appveyor CI have to be uploaded to PyPI. They can be found on the appveyor
CI run for the created tag under the ``jobs/Artifacts`` tab. All the ``.whl``
files should be downloaded into a folder. They can then be added to the release
on PyPI using e.g. ``twine upload pytesmo-0.7.1*whl``

Note
====

This project has been set up using PyScaffold 3.2.3. For details and usage
information on PyScaffold see https://pyscaffold.org/.
