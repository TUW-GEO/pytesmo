# -*- coding: utf-8 -*-
"""
    Setup file for pytesmo.
    Use setup.cfg to configure your project.

    This file was generated with PyScaffold 3.2.3.
    PyScaffold helps you to put up the scaffold of your new Python project.
    Learn more under: https://pyscaffold.org/
"""

import pkg_resources
from pkg_resources import VersionConflict, require
from setuptools import setup
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.develop import develop
from setuptools.command.install import install
from setuptools.extension import Extension
import sys


def custom_cythonize():
    from Cython.Build import cythonize

    cythonize(
        [
            "src/pytesmo/time_series/filters.pyx",
        ],
    )


# cythonize when ``--cythonize`` command line option is passed
class CythonizeMixin:

    user_options = [
        ("cythonize", None, "recreate the c extionsions with cython")
    ]

    def initialize_options(self):
        super().initialize_options()
        self.cythonize = False

    def run(self):
        if self.cythonize:
            custom_cythonize()
        super().run()


class develop_with_cythonize(CythonizeMixin, develop):
    user_options = (
        getattr(develop, "user_options", []) + CythonizeMixin.user_options
    )


class install_with_cythonize(CythonizeMixin, install):
    user_options = (
        getattr(install, "user_options", []) + CythonizeMixin.user_options
    )


class build_ext(_build_ext):

    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


ext_modules = [
    Extension(
        "pytesmo.time_series.filters",
        ["src/pytesmo/time_series/filters.c"],
    ),
]


try:
    require("setuptools>=38.3")
except VersionConflict:
    print("Error: version of setuptools is too old (<38.3)!")
    sys.exit(1)


if __name__ == "__main__":
    cmdclass = {}
    cmdclass["build_ext"] = build_ext
    cmdclass["develop"] = develop_with_cythonize
    cmdclass["install"] = install_with_cythonize
    setup(use_pyscaffold=True, cmdclass=cmdclass, ext_modules=ext_modules)
