"""
This test module runs code from docs/examples ipynbs to make sure that the
documentation is up-to-date with the package, resp. that the notebooks will
build correctly on readthedocs.
"""

import os
import subprocess
from nbconvert.preprocessors import ExecutePreprocessor
import nbformat
import pytest
from pytesmo.utils import rootdir

examples_path = os.path.join(rootdir(), 'docs', 'examples')

@pytest.mark.parametrize("notebook", [
    "anomalies.ipynb",
    "compare_ASCAT_ISMN.ipynb",
    "swi_calc.ipynb",
    "temporal_collocation.ipynb",
    "triple_collocation.ipynb",
    "validation_framework.ipynb",
])
@pytest.mark.skipif(
    not os.path.isdir(examples_path),
    reason=f"Directory '{examples_path}' not found. "
           "Pytesmo is probably not installed in `editable` mode."
)
@pytest.mark.slow
@pytest.mark.doc_example
def test_ipython_notebook(notebook):
    """
    Run selected ipynb example files from docs/examples as tests.
    IMPORTANT: In order that the paths to testdata from notebooks are also
    applicable to the tests here, this file must be within a sub-folder of
    the tests/ directory (assuming that examples are in docs/examples)!
    """
    # Handles jupyter warning (can probably be removed again in future):
    os.environ["JUPYTER_PLATFORM_DIRS"] = "1"
    subprocess.call(["jupyter", "--paths"])

    # Run ipynb files and check if they pass
    preprocessor = ExecutePreprocessor(timeout=600, kernel_name="python3")
    with open(os.path.join(examples_path, notebook)) as f:
        nb = nbformat.read(f, as_version=4)
    preprocessor.preprocess(nb)
