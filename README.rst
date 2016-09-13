=========================================================================================
Introduction to pytesmo a Python Toolbox for the Evaluation of Soil Moisture Observations
=========================================================================================
.. image:: https://travis-ci.org/TUW-GEO/pytesmo.svg?branch=master
    :target: https://travis-ci.org/TUW-GEO/pytesmo

.. image:: https://coveralls.io/repos/TUW-GEO/pytesmo/badge.png?branch=master
  :target: https://coveralls.io/r/TUW-GEO/pytesmo?branch=master

.. image:: https://badge.fury.io/py/pytesmo.svg
    :target: http://badge.fury.io/py/pytesmo


pytesmo is a package which aims it is to provide a library that can be used for the comparison and validation
of geospatial time series datasets with a (initial) focus on soil moisture.

Documentation
=============

Please see the latest documentation including examples of how to install and use pytesmo
at http://pytesmo.readthedocs.io/ .

Citation
========

If you use the software please cite it. Until our paper is finished please use
the Zenodo DOI below:

.. image:: https://zenodo.org/badge/12761/TUW-GEO/pytesmo.svg
   :target: https://zenodo.org/badge/latestdoi/12761/TUW-GEO/pytesmo

Contribute
==========

We are happy if you want to contribute. Please raise an issue explaining what is missing
or if you find a bug. We will also gladly accept pull requests against our master branch
for new features or bug fixes.

If you want to contribute please follow these steps:

- Fork the pytesmo repository to your account
- Clone the repository, make sure you use ``git clone --recursive`` to also get
  the test data repository.
- make a new feature branch from the pytesmo master branch
- hack away
- please include tests for your contributions in one of the test directories
  We use py.test so a simple function called test_my_feature is enough
- submit a pull request to our master branch
