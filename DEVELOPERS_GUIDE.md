Developers Guide
================

Setup
-----

1) Clone your fork of pytesmo to your machine using ``git clone --recursive``
   to also get the test data
2) Create a new conda environment:
    ```
    conda env create -f environment.yml
    conda activate pytesmo
    ```

3) Install pytesmo for development:
    ```
    pip install -e .
    ```
   (Note the dot at the end of the command). This is the recommended way to
   install the package and should be preferred over any `python setup.py`
   commands (e.g. `python setup.py develop` or `python setup.py install`). [See
   below](#setup.py-commands) for more info.

4) Optional: Install the pre-commit hooks:
    ```
    pre-commit install
    ```
   This runs a few checks before you commit your code to make sure it's nicely
   formatted.


Now you should be ready to go.


Adding a feature
----------------

To add a new feature, you should create a new branch on your fork. First, make
sure that the master branch of your fork is up to date with
`TUW-GEO/pytesmo`. To do this, you can add `TUW-GEO/pytesmo` as a new remote:
    ```
    git remote add upstream git@github.com:TUW-GEO/pytesmo.git
    ```
Then you can fetch the upstream changes and merge them into your master. Don't
forget to push your local master to your Github fork:
    ```
    git fetch upstream
    git merge upstream/master --ff
    git push
    ```
Create a local branch:
    ```
    git checkout -b my_feature_branch
    ```

Now add your feature. Please add some tests in the test directory. See below for
how to run them. Once you are done, you can add and commit your changes (`git
add <changed files>` and `git commit`) and  push them to your fork. The first
time you push, you have to do
    ```
    git push -u origin my_feature_branch
    ```
Afterwards, `git push` is enough.

Once your done you can open a pull request on Github.

Code Formatting
---------------
To apply pep8 conform styling to any changed files [we use `yapf`](https://github.com/google/yapf). The correct
settings are already set in `setup.cfg`. Therefore the following command
should be enough:

    yapf file.py --in-place

Testing
-------

Tests can (and should) be run with the command `pytest`. The old command
`python setup.py test` [is deprecated](https://github.com/pypa/setuptools/issues/1684).


Working with Cython
-------------------

In case you change something in the cython extensions, make sure to run:

    python setup.py build_ext --inplace --cythonize

after you applied your changes. There will be some warnings like this:

    warning: src/pytesmo/metrics/_fast.pyx:11:6: Unused entry 'dtype_signed'

Ignore the warnings about unused entries, e.g. 'dtype_signed', 'itemsize',
'kind', 'memslice', 'ndarray', and maybe some more. If there are other warnings
than unused entry, or if one of the unused entries looks like a variable name
you used, you should probably investigate them.

Remember to check in the generated C-files, because the built binary packages
uploaded to PyPI will be based on those.


Setup.py commands
-----------------

PEP 517 tries to unify the way how Python packages are built and does not
require a `setup.py` anymore (it uses `pyproject.toml` instead).
Therefore, commands like `python setup.py install`, `python setup.py develop`,
`python setup.py sdist`, and `python setup.py bdist_wheel` are not necessarily
supported anymore. Therefore, we want to encourage everyone to use more
future-proof commands:

- `pip install .` instead of `python setup.py install`
- `pip install -e .` instead of `python setup.py develop`
- `pytest` instead of `python setup.py test`
- `python -m build --sdist` instead of `python setup.py sdist` (needs `build` installed)
- `python -m build --wheel` instead of `python setup.py bdist_wheel` (needs `build` installed)

Due to our more complex build setup (using Cython) we still need a `setup.py`,
and the old commands will still work (except `python setup.py test`), but we
recommend to use the new commands.


Creating a release
------------------

To release a new version of this package, make sure all tests are passing
on the master branch and the `CHANGELOG.rst` is up-to-date, with changes for
the new version at the top.

Then draft a new release on [GitHub](https://github.com/TUW-GEO/pytesmo/releases).
Create a version tag following the ``v{MAJOR}.{MINOR}.{PATCH}`` pattern.
This will trigger a new build on GitHub and should push the packages to pypi after
all tests have passed.

If this does not work (tests pass but upload fails) you can download the
``whl`` and ``dist`` packages for each workflow run from
https://github.com/TUW-GEO/pytesmo/actions (Artifacts) and push them manually to
https://pypi.org/project/pytesmo/ e.g. using [twine](https://pypi.org/project/twine/)
(you need to be a package maintainer on pypi for that).
