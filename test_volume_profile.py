import asyncio

import numpy as np
import pandas as pd

from quantist_library.helper import Bin


def _peaks(values: list[float]) -> list[int]:
    return asyncio.run(Bin(pd.DataFrame()).calc_peaks_index(pd.Series(values)))


def test_volume_profile_significance_excludes_weak_local_extrema():
    histogram = [0.0, 100.0, 0.0, 12.0, 10.0, 12.0, 0.0]

    assert _peaks(histogram) == [1]


def test_volume_profile_keeps_a_dominant_negative_valley():
    histogram = [0.0, -5.0, 0.0, -100.0, 0.0]

    assert _peaks(histogram) == [3]


def test_volume_profile_keeps_sign_separators_in_peak_mapping():
    histogram = [0.0, 100.0, -1.0, 90.0, 0.0]

    assert _peaks(histogram) == [1, 3]


def test_volume_profile_falls_back_to_dominant_node_without_local_extrema():
    histogram = [5.0, 10.0, 15.0]

    assert _peaks(histogram) == [2]


def test_volume_profile_fit_handles_zero_range_and_zero_calculated_bins():
    data = pd.DataFrame({
        "close": np.array([100.0, 100.0, 100.0]),
        "netval": np.array([10.0, -5.0, 2.0]),
    })

    fitted = asyncio.run(Bin(data).fit())

    assert fitted.nbins == 1
    assert len(fitted.hist_bar) == 1
    assert fitted.peaks_index == [0]
