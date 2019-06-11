__all__ = ['metrics', 'scaling', 'temporal_matching',
           'timedate', 'time_series',
           'grid', 'io', 'colormaps']

import pkg_resources

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except:
    __version__ = 'unknown'
