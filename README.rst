=======
pytesmo
=======
.. image:: https://travis-ci.org/TUW-GEO/pytesmo.svg?branch=master
    :target: https://travis-ci.org/TUW-GEO/pytesmo

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

Citation
========

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.596422.svg
   :target: https://doi.org/10.5281/zenodo.596422

If you use the software in a publication then please cite it using the Zenodo DOI.
Be aware that this badge links to the latest package version.

Please select your specific version at https://doi.org/10.5281/zenodo.596422 to get the DOI of that version.
You should normally always use the DOI for the specific version of your record in citations.
This is to ensure that other researchers can access the exact research artefact you used for reproducibility.

You can find additional information regarding DOI versioning at http://help.zenodo.org/#versioning

Installation
============

This package should be installable through pip:

.. code::

    pip install pytesmo

Supported Products
==================

Soil moisture is observed using different methods and instruments, in this
version several satellite datasets as well as in situ data are supported:

- ASCAT
- H-SAF image products
- SMAP
- GLDAS Noah
- ERA Interim
- ERS
- Data from the International Soil Moisture Network (ISMN)

For more details visit the Documentation_.

.. _Documentation: https://pytesmo.readthedocs.io/en/latest/?badge=latest

Contribute
==========

We are happy if you want to contribute. Please raise an issue explaining what
is missing or if you find a bug. We will also gladly accept pull requests
against our master branch for new features or bug fixes.

Development setup
-----------------

For Development we recommend a ``conda`` environment.

Guidelines
----------

If you want to contribute please follow these steps:

- Fork the pytesmo repository to your account
- Clone the repository, make sure you use ``git clone --recursive`` to also get
  the test data repository.
- make a new feature branch from the pytesmo master branch
- Add your feature
- please include tests for your contributions in one of the test directories
  We use py.test so a simple function called test_my_feature is enough
- submit a pull request to our master branch
