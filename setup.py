from distutils.core import setup

setup(
    name='pytesmo',
    version='0.1.1',
    author='pytesmo Team',
    author_email='Christoph.Paulik@geo.tuwien.ac.at',
    packages=['pytesmo','pytesmo.timedate','pytesmo.grid','pytesmo.io','pytesmo.io.sat','pytesmo.io.ismn'],
    scripts=['bin/plot_ASCAT_data.py','bin/plot_ISMN_data.py','bin/compare_ISMN_ASCAT.py'],
    url='http://rs.geo.tuwien.ac.at/validation_tool/pytesmo/',
    license='LICENSE.txt',
    description='python Toolbox for the Evaluation of Soil Moisture Observations',
    long_description=open('README.txt').read(),
    install_requires=[
        "numpy >= 1.7.1",
        "pandas >= 0.11.0",
        "scipy >= 0.12.0",
        "statsmodels >= 0.4.3",
    ],
)