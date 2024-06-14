"""
    Setup file for pytesmo.
    Use setup.cfg to configure your project.

    This file was generated with PyScaffold 4.5.
    PyScaffold helps you to put up the scaffold of your new Python project.
    Learn more under: https://pyscaffold.org/
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
        Extension(
            "pytesmo.metrics._fast_pairwise",
            ["src/pytesmo/metrics/_fast_pairwise" + ext],
            include_dirs=[numpy.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        ),
    ]


# defining a custom cythonize function that sets all the options we want and
# that can be reused for the different options
def cythonize_extensions():
    from Cython.Build import cythonize

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


class CythonizeMixin(object):

    user_options = [
        ("cythonize", None, "recreate the c extionsions with cython")
    ]

    def initialize_options(self):
        super().initialize_options()
        self.cythonize = False

    def run(self):
        if self.cythonize:
            cythonize_extensions()
        super().run()


class sdist(CythonizeMixin, _sdist):
    user_options = getattr(_sdist, 'user_options', [])\
               + CythonizeMixin.user_options


class build_ext(CythonizeMixin, _build_ext):
    user_options = getattr(_build_ext, 'user_options', [])\
               + CythonizeMixin.user_options


if __name__ == "__main__":
    try:
        setup(use_scm_version={"version_scheme": "no-guess-dev"})
    except:  # noqa
        print(
            "\n\nAn error occurred while building the project, "
            "please ensure you have the most updated version of setuptools, "
            "setuptools_scm and wheel with:\n"
            "   pip install -U setuptools setuptools_scm wheel\n\n"
        )
        raise
