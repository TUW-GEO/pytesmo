pytesmo - a Python Toolbox for the Evaluation of Soil Moisture Observations
***************************************************************************

|ci| |cov| |pip| |doc|

.. |ci| image:: https://github.com/TUW-GEO/pytesmo/actions/workflows/ci.yml/badge.svg?branch=master
   :target: https://github.com/TUW-GEO/pytesmo/actions

.. |cov| image:: https://coveralls.io/repos/TUW-GEO/pytesmo/badge.png?branch=master
  :target: https://coveralls.io/r/TUW-GEO/pytesmo?branch=master

.. |pip| image:: https://badge.fury.io/py/pytesmo.svg
    :target: https://badge.fury.io/py/pytesmo

.. |doc| image:: https://readthedocs.org/projects/pytesmo/badge/?version=latest
   :target: https://pytesmo.readthedocs.io/en/latest/

pytesmo, the Python Toolbox for the Evaluation of Soil Moisture Observations, is
a package/python toolbox which aims to provide a library that can be used for
the comparison and validation of geospatial time series datasets with a
(initial) focus on soil moisture.

Documentation & Software Citation
=================================

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.596422.svg
   :target: https://doi.org/10.5281/zenodo.596422

To see the latest `full documentation <https://pytesmo.readthedocs.io/en/latest/?badge=latest>`_
click on the docs badge at the top.

If you use the software in a publication then please cite it using the Zenodo
DOI.  Be aware that this badge links to the latest package version.

Please select your specific version at https://doi.org/10.5281/zenodo.596422 to
get the DOI of that version.  You should normally always use the DOI for the
specific version of your record in citations.  This is to ensure that other
researchers can access the exact research artefact you used for
reproducibility.

You can find additional information regarding DOI versioning at
http://help.zenodo.org/#versioning

If you want to contribute, take a look at the `developers guide
<https://github.com/TUW-GEO/pytesmo/blob/master/DEVELOPERS_GUIDE.md>`_ .

Installation
============

This package can be installed through `pip
<https://pip.pypa.io/en/latest/installing.html>`__ which downloads the package
from the `python package repository Pypi <https://pypi.python.org/>`__.

.. code-block:: bash

    pip install pytesmo

Note that ``pytesmo>=0.17`` will only work with ``numpy>=2.0`` and ``python>=3.9``.
If you still require a version compatible with numpy v1, install ``pytesmo<0.17``.

Compiled C extensions are available as wheels for Windows and Linux for
python 3.9 to 3.12.

For other operating systems and/or python versions, it might be necessary
to compile the binaries yourself. This should
happen automatically, if you have a C compiler installed (e.g.
`GCC <https://gcc.gnu.org/>`_ ).

.. code::

    git clone https://github.com/TUW-GEO/pytesmo.git --recursive
    cd pytesmo
    mamba create -n pytesmo python=3.12 # or any supported python version
    conda activate pytesmo
    pip install -e .[testing]  # install source package
    python setup.py build_ext --inplace --cythonize  # optional compile cython C extensions
    pytest  # Run tests to check if everything works

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

Related Packages
================

Some former pytesmo modules are now provided as separate packages.

- `pygeogrids <https://github.com/TUW-GEO/pygeogrids/>`_ : Creation and handling of Discrete Global Grids or Point collections
- `cadati <https://github.com/TUW-GEO/cadati/>`_ : Calender, Date and Time functions
- `repurpose <https://github.com/TUW-GEO/repurpose/>`_ : Time series - image conversion and resampling routines
- `colorella <https://github.com/TUW-GEO/colorella/>`_ : Color maps and color map handling


Contribute
==========

We are happy if you want to contribute. Please raise an issue explaining what
is missing or if you find a bug. We will also gladly accept pull requests
against our master branch for new features or bug fixes.

Please follow the `developers guide
<https://github.com/TUW-GEO/pytesmo/blob/master/DEVELOPERS_GUIDE.md>`_.
