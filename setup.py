try:
    from setuptools import setup
    have_setuptools = True
    from setuptools.command.test import test as TestCommand
except ImportError:
    have_setuptools = False
    from distutils.core import setup
from distutils.extension import Extension
import numpy as np
from distutils.command.sdist import sdist as _sdist
import sys
import os


class sdist(_sdist):
    def run(self):
        # Make sure the compiled Cython files in the distribution are up-to-date
        from Cython.Build import cythonize
        cythonize(['pytesmo/time_series/filters.pyx'])
        _sdist.run(self)
cmdclass = {}
cmdclass['sdist'] = sdist


ext_modules = [
    Extension("pytesmo.time_series.filters", ["pytesmo/time_series/filters.c"],
              include_dirs=[np.get_include()]),
]

if not have_setuptools:
    setuptools_kwargs = {}
else:
    class PyTest(TestCommand):
        def finalize_options(self):
            TestCommand.finalize_options(self)
            self.test_args = []
            self.test_suite = True

        def run_tests(self):
            import pytest
            errcode = pytest.main(self.test_args)
            sys.exit(errcode)

    cmdclass['test'] = PyTest
    setuptools_kwargs = {'install_requires': ["numpy >= 1.7",
                                              "pandas >= 0.12",
                                              "scipy >= 0.12",
                                              "statsmodels >= 0.4.3",
                                              "netcdf4 >= 1.0.1",
                                           ],
                         'test_suite': 'tests/',
                         'tests_require': ['pytest'],
                         'extras_require': {'testing': ['pytest']
                                            }
                       }


setup(
    name='pytesmo',
    version='0.2.1',
    author='pytesmo Team',
    author_email='Christoph.Paulik@geo.tuwien.ac.at',
    packages=['pytesmo', 'pytesmo.timedate',
              'pytesmo.grid', 'pytesmo.io', 'pytesmo.io.sat', 'pytesmo.io.ismn',
              'pytesmo.io.bufr', 'pytesmo.colormaps',
              'pytesmo.time_series', 'pytesmo.timedate'],
    ext_modules=ext_modules,
    package_data={'pytesmo': [os.path.join('colormaps', '*.cmap')],
                 },
    cmdclass=cmdclass,
    url='http://rs.geo.tuwien.ac.at/validation_tool/pytesmo/',
    license='LICENSE.txt',
    description='python Toolbox for the Evaluation of Soil Moisture Observations',
    long_description=open('README.rst').read(),
    **setuptools_kwargs)
