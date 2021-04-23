Developers Guide
================

Setup
-----

1) Clone your fork of pytesmo to your machine using ``git clone --recursive``
   to also get the test data
2) Create a new conda environment:

     conda env create -f environment.yml
     conda activate pytesmo

3) Install the pre-commit hooks:

     pre-commit install

   This runs a few checks before you commit your code to make sure it's nicely
   formatted.

Now you should be ready to go.

Cython
------

In case you change something in the cython extensions, make sure to run:

    python setup.py build_ext --inplace --cythonize

after you applied your changes. There will be some warnings like this:

    warning: src/pytesmo/metrics/_fast.pyx:11:6: Unused entry 'dtype_signed'

Ignore the warnings about unused entries, e.g. 'dtype_signed', 'itemsize',
'kind', 'memslice', 'ndarray', and maybe some more. If there are other warnings
than unused entry, or if one of the unused entries looks like a variable name
you used, you should probably investigate them.
