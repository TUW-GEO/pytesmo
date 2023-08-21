import numpy as np
import pandas as pd


def create_correlated_data(n_datasets, n, r):
    """Creates n_datasets random timeseries with specified correlation"""
    C = np.ones((n_datasets, n_datasets)) * r
    for i in range(n_datasets):
        C[i, i] = 1
    A = np.linalg.cholesky(C)

    return (A @ np.random.randn(n_datasets, n)).T


class DummyGrid:
    # required for the dummy reader
    def find_nearest_gpi(self, lon, lat, max_dist=np.inf):
        return 0, 0


class DummyReader:

    def __init__(self, dfs, name):
        if not isinstance(dfs, (list, tuple)):
            dfs = [dfs]
        self.data = [pd.DataFrame(dfs[i][name]) for i in range(len(dfs))]
        self.grid = DummyGrid()

    def read(self, gpi, *args, **kwargs):
        return self.data[gpi]


class DummyEmptyDataFrameReader:
    # has data, but only an empty dataframe

    def __init__(self, dfs, name):
        self.data = [pd.DataFrame(dfs[i][name]) for i in range(len(dfs))]
        self.grid = DummyGrid()

    def read(self, gpi, *args, **kwargs):
        names = self.data[gpi].columns
        return pd.DataFrame(np.zeros((0, len(names))), columns=names)


class DummyNoDataReader:
    # has no data at all
    def __init__(self, dfs, name):
        self.grid = DummyGrid()

    def read(self, gpi, *args, **kwargs):
        return []


def create_datasets(n_datasets, npoints, nsamples, missing=False):
    """
    Creates three datasets with given number of points to compare, each
    having number of samples given
    """
    dfs = []
    for gpi in range(npoints):
        r = np.random.rand()
        data = create_correlated_data(n_datasets, nsamples, r)
        index = pd.date_range("1980", periods=nsamples, freq="D")
        dfs.append(
            pd.DataFrame(
                data,
                index=index,
                columns=(["refcol"] +
                         [f"other{i}col" for i in range(1, n_datasets)])))

    datasets = {}
    datasets["0-ERA5"] = {
        "columns": ["refcol"],
        "class": DummyReader(dfs, "refcol")
    }
    for i in range(1, n_datasets - 1):
        datasets[f"{i}-ESA_CCI_SM_combined"] = {
            "columns": [f"other{i}col"],
            "class": DummyReader(dfs, f"other{i}col")
        }
    if missing == "empty":
        datasets[f"{n_datasets - 1}-missing"] = {
            "columns": [f"other{n_datasets - 1}col"],
            "class": DummyEmptyDataFrameReader(dfs, f"other{n_datasets - 1}col")
        }
    elif missing == "nodata":
        datasets[f"{n_datasets - 1}-missing"] = {
            "columns": [f"other{n_datasets - 1}col"],
            "class": DummyNoDataReader(dfs, f"other{n_datasets - 1}col")
        }
    else:
        datasets[f"{n_datasets - 1}-ESA_CCI_SM_combined"] = {
            "columns": [f"other{n_datasets - 1}col"],
            "class": DummyReader(dfs, f"other{n_datasets - 1}col")
        }
    return datasets
