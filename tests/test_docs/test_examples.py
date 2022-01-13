"""
This test module runs code from docs/examples ipynbs to make sure that the
documentation is up-to-date with the package, resp. that the docs will build
correctly.
"""

import os
from nbconvert.preprocessors import ExecutePreprocessor
import nbformat
import pytest


@pytest.mark.parametrize("notebook", [
    "anomalies.ipynb",
    "compare_ASCAT_ISMN.ipynb",
    "swi_calc.ipynb",
    "temporal_collocation.ipynb",
    "triple_collocation.ipynb",
    "validation_framework.ipynb",
])
def test_ipython_notebook(notebook):
    """
    Run selected ipynb example files from docs/examples as tests.
    IMPORTANT: In order that the paths to testdata from notebooks are also
    applicable to the tests here, this file must be within a sub-folder of
    the tests/ directory (assuming that the examples in docs/examples)!
    """
    examples_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'docs', 'examples')
    preprocessor = ExecutePreprocessor(timeout=600, kernel_name="python3")
    with open(os.path.join(examples_path, notebook)) as f:
        nb = nbformat.read(f, as_version=4)
    preprocessor.preprocess(nb)
