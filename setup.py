# -*- coding: utf-8 -*-
"""
    Setup file for pytesmo.
    Use setup.cfg to configure your project.

    This file was generated with PyScaffold 3.2.3.
    PyScaffold helps you to put up the scaffold of your new Python project.
    Learn more under: https://pyscaffold.org/

    The code to cythonize extensions was added manually.
"""

from setuptools import setup
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.sdist import sdist as _sdist
from setuptools.extension import Extension
import numpy


# list of C/Cython extensions
def get_ext_modules(ext):
    return [
        Extension(
            "pytesmo.time_series.filters",
            ["src/pytesmo/time_series/filters" + ext],
            include_dirs=[numpy.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        ),
        Extension(
            "pytesmo.metrics._fast",
            ["src/pytesmo/metrics/_fast" + ext],
            include_dirs=[numpy.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        ),
    ]


# defining a custom cythonize function that sets all the options we want and
# that can be reused for the different options
def cythonize_extensions():
    from Cython.Build import cythonize
    import numpy
    cythonize(
        get_ext_modules(".pyx"),
        compiler_directives={
            "embedsignature": True,
            "language_level": 3,
            # "warn.undeclared": True,
            # "warn.maybe_unitialized": True,
            "warn.unused": True,
            # "warn.unused_arg": True,
            # "warn.unused_result": True,
        },
        # include_path=[numpy.get_include()],
    )


# We want to cythonize the .pyx modules (that is, regenerate the .c files)
# whenever we run build_ext or sdist, so we always ship up to date .c files.
# Therefore we subclass the setuptools versions of those and tell them to use
# Cython
class sdist(_sdist):
    def run(self):
        cythonize_extensions()
        super().run()


class build_ext(_build_ext):
    def run(self):
        cythonize_extensions()
        super().run()


if __name__ == '__main__':
    cmdclass = {}
    cmdclass["build_ext"] = build_ext
    cmdclass["sdist"] = sdist
    setup(
        cmdclass=cmdclass,
        # at this point the C modules have already been generated if necessary
        ext_modules=get_ext_modules(".c")
    )
