"""
An empty window has no bars, and saying so must not raise.

``int(frame.groupby(level="code").size().max())`` is ``int(NaN)`` on a frame
with no rows — a ``ValueError`` three call levels below the request. An empty
universe (no broker past the criterion, no rows inside the window) reached
exactly that, right beside the vprofile ``KeyError`` the same run hit first. A
window with no rows has no bars: the screener reading it has no candidates, and
both answer empty instead of failing.
"""
import datetime

import pandas as pd
import pytest

from quantist_library.helper import bar_range_of


def _window(per_code: dict[str, int]) -> pd.DataFrame:
	"""One (code, date) indexed frame holding `per_code` bars for each code."""
	entries = [
		(code, datetime.date(2026, 9, 30) - datetime.timedelta(days=offset))
		for code, bars in per_code.items()
		for offset in range(bars)
	]
	index = pd.MultiIndex.from_tuples(entries, names=["code", "date"])
	return pd.DataFrame({"close": [100.0] * len(entries)}, index=index)


def test_the_busiest_code_sets_the_window_length():
	assert bar_range_of(_window({"BBCA": 5, "TLKM": 3})) == 5


def test_a_window_widened_for_a_single_code_reports_that_codes_bars():
	assert bar_range_of(_window({"BBCA": 7})) == 7


def test_an_empty_window_reports_no_bars_instead_of_raising():
	# The shape the screeners hand over after a selection comes back empty:
	# every column still there, not one row.
	assert bar_range_of(_window({"BBCA": 3}).iloc[0:0]) == 0


def test_a_frame_that_never_had_the_code_index_still_raises():
	# 0 is the answer for "no rows", not for "wrong frame": reporting no bars
	# for a frame the callers cannot read would hide the mistake.
	with pytest.raises(ValueError, match="level name code"):
		bar_range_of(pd.DataFrame({"close": [100.0]}))
