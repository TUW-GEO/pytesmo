import numpy as np
from pytesmo.interpolate.dctpls import smoothn

np.random.seed(123)


def make_data_3d(n=100, exponent=2, center_gap=True) -> np.ndarray:
    """
    Exponentially increasing 3D datasets towards the center of the cube
    """
    x, y, z = np.meshgrid(
        np.arange(n), np.arange(n), np.arange(n), indexing="ij"
    )

    center = (n - 1) / 2
    distances = np.sqrt((x - center) ** 2 + (y - center) ** 2 +
                        (z - center) ** 2)
    distances_normalized = distances / distances.max()
    exponential_values = 1 / distances_normalized ** exponent

    if center_gap:
        s = slice(int(n / 2) - 5, int(n / 2) + 5)
        exponential_values[s, s, s] = np.nan

    return exponential_values


def test_dctpls_synth_data_1d():
    data = (np.sin(np.linspace(0, 2 * np.pi, 100)) +
            np.random.rand(100))
    data[50:60] = np.nan
    exclusion_mask = np.array([True] +
                              np.full(99, False).tolist())

    data_smoothed, s, flag, stats = \
        smoothn(data, exclusion_mask=exclusion_mask)

    assert np.isnan(data_smoothed[0])
    np.testing.assert_almost_equal(s, np.array([483.1286967]))
    np.testing.assert_almost_equal(float(data_smoothed[55]),
                                   0.267432, 5)
    np.testing.assert_almost_equal(float(data_smoothed[10]),
                                   1.043243, 5)
    np.testing.assert_almost_equal(float(data_smoothed[80]),
                                   -0.413490, 5)


def test_dctpls_synth_data_2d():
    data = make_data_3d(50, 2, center_gap=True)[25, :, :]
    data_smoothed, s, flag, stats = smoothn(data)
    np.testing.assert_almost_equal(s, np.array([1.7333483]))

    assert not np.any(np.isnan(data_smoothed))
    np.testing.assert_almost_equal(float(data_smoothed[3, 3]),
                                   1.928477, 5)
    np.testing.assert_almost_equal(float(data_smoothed[25, 25]),
                                   37.990380, 5)
    np.testing.assert_almost_equal(float(data_smoothed[47, 47]),
                                   1.7554522, 5)


def test_dctpls_synth_data_3d():
    data = make_data_3d(50, 2, center_gap=True)
    return_stats = ('initial_guess', 'euclidean_distance',
                    'final_weights', 'gcv-score')
    data_smoothed, s, flag, stats = smoothn(data, return_stats=return_stats)
    np.testing.assert_almost_equal(s, np.array([0.6171467]))
    assert not np.any(np.isnan(data_smoothed))
    np.testing.assert_almost_equal(
        float(data_smoothed[25, 25, 25]), 45.33357123, 5)
    np.testing.assert_almost_equal(
        float(data_smoothed[3, 3, 3]), 1.2932548, 5)
    np.testing.assert_almost_equal(
        float(data_smoothed[47, 47, 47]), 1.1756009, 5)
