try:
    from setuptools import setup
    have_setuptools = True
except ImportError:
    have_setuptools = False
    from distutils.core import setup
from distutils.extension import Extension
import numpy as np
from distutils.command.sdist import sdist as _sdist


class sdist(_sdist):
    def run(self):
        # Make sure the compiled Cython files in the distribution are up-to-date
        from Cython.Build import cythonize
        cythonize(['pytesmo/time_series/filters.pyx'])
        _sdist.run(self)
cmdclass = {}
cmdclass['sdist'] = sdist


ext_modules = [
    Extension("pytesmo.time_series.filters", [ "pytesmo/time_series/filters.c" ],
              include_dirs=[np.get_include()]),
]



if not have_setuptools:
    setuptools_kwargs = {}
else:
    setuptools_kwargs = {'install_requires':[ "numpy >= 1.7.1",
                                            "pandas >= 0.11.0",
                                            "scipy >= 0.12.0",
                                            "statsmodels >= 0.4.3",
                                            "netcdf4 >= 1.0.1",
                                           ]
                       }

setup(
    name='pytesmo',
    version='0.1.1',
    author='pytesmo Team',
    author_email='Christoph.Paulik@geo.tuwien.ac.at',
    packages=['pytesmo', 'pytesmo.timedate', 'pytesmo.grid', 'pytesmo.io', 'pytesmo.io.sat', 'pytesmo.io.ismn',
              'pytesmo.time_series', 'pytesmo.timedate'],
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    scripts=['bin/plot_ASCAT_data.py', 'bin/plot_ISMN_data.py', 'bin/compare_ISMN_ASCAT.py',
             'bin/read_ASCAT_H25.py', 'bin/anomalies.py'],
    url='http://rs.geo.tuwien.ac.at/validation_tool/pytesmo/',
    license='LICENSE.txt',
    description='python Toolbox for the Evaluation of Soil Moisture Observations',
    long_description=open('README.txt').read(),
    **setuptools_kwargs)
