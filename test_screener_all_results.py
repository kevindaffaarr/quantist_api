"""
Unbounded screener results.

The web surface caches a criterion's *whole* answer; the Telegram commands
want the top ten. Both run through the same rank helpers, so the helpers grew
an explicit `None` meaning "every candidate" — not a large number standing in
for all, which is the same bug one order of magnitude later.

Truncation lives in four places across three families plus one SQL LIMIT, and
these pin every one of them.
"""
import inspect
import pathlib

import pandas as pd
import pytest
from sqlalchemy import select

import database as db
import dependencies as dp
from quantist_library import foreignflow as ff
from quantist_library import screener as sc

VWAP_CRITERIA = [
	dp.ScreenerList.vwap_rally,
	dp.ScreenerList.vwap_around,
	dp.ScreenerList.vwap_breakout,
	dp.ScreenerList.vwap_breakdown,
]
VPROFILE_CRITERIA = [
	dp.ScreenerList.vprofile_inside,
	dp.ScreenerList.vprofile_breakout,
	dp.ScreenerList.vprofile_breakdown,
	dp.ScreenerList.vprofile_support_bounce,
	dp.ScreenerList.vprofile_resistance_rejection,
]

ROWS = 900


def _codes(count: int = ROWS) -> list[str]:
	return [f"s{index:04d}" for index in range(count)]


def _flow_frame(count: int = ROWS) -> pd.DataFrame:
	return pd.DataFrame({"mf": [float(count - index) * 1e9 for index in range(count)]}, index=pd.Index(_codes(count), name="code"))


def _vwap_data(count: int = ROWS) -> pd.DataFrame:
	"""Two bars per code, so the cross helpers have a previous bar to look at."""
	frames = []
	for index, code in enumerate(_codes(count)):
		frames.append(pd.DataFrame({
			"close": [100.0 + index, 101.0 + index],
			"vwap": [99.0 + index, 100.0 + index],
			"netval": [1.0e9, 2.0e9],
		}, index=pd.MultiIndex.from_product([[code], pd.to_datetime(["2026-09-17", "2026-09-18"])], names=["code", "date"])))
	return pd.concat(frames)


def _vprofile_frame(count: int = ROWS) -> pd.DataFrame:
	return pd.DataFrame({
		"vprofile_zone_prominence": [float(count - index) for index in range(count)],
		"vprofile_zone_strength": [0.5] * count,
		"mf": [float(index) * 1e8 for index in range(count)],
	}, index=pd.Index(_codes(count), name="code"))


# ==========
# Money flow
# ==========
def test_flow_ranking_returns_every_candidate_when_unbounded():
	ranked = sc.rank_flow_candidates(_flow_frame(), None)
	assert len(ranked) == ROWS


def test_flow_ranking_still_truncates_for_the_legacy_default():
	assert len(sc.rank_flow_candidates(_flow_frame(), 10)) == 10


def test_flow_ranking_unbounded_keeps_the_same_order_as_bounded():
	full = sc.rank_flow_candidates(_flow_frame(), None)
	top = sc.rank_flow_candidates(_flow_frame(), 10)
	# The first ten of "everything" are exactly the bounded ten, in order.
	assert full.index[:10].tolist() == top.index.tolist()


@pytest.mark.parametrize("ascending", [True, False])
def test_flow_ranking_is_deterministic_in_both_directions(ascending):
	first = sc.rank_flow_candidates(_flow_frame(), None, ascending=ascending)
	second = sc.rank_flow_candidates(_flow_frame(), None, ascending=ascending)
	assert first.index.tolist() == second.index.tolist()
	assert len(first) == ROWS


# ==========
# VWAP
# ==========
@pytest.mark.parametrize("criteria", VWAP_CRITERIA)
def test_vwap_ranking_returns_every_candidate_when_unbounded(criteria):
	data = _vwap_data()
	ranked = sc.rank_vwap_candidates(data, _codes(), None, criteria)
	assert len(ranked) == ROWS
	assert len(set(ranked)) == ROWS


@pytest.mark.parametrize("criteria", VWAP_CRITERIA)
def test_vwap_ranking_still_truncates_for_the_legacy_default(criteria):
	assert len(sc.rank_vwap_candidates(_vwap_data(), _codes(), 10, criteria)) == 10


@pytest.mark.parametrize("criteria", VWAP_CRITERIA)
def test_vwap_unbounded_starts_with_the_bounded_result(criteria):
	data = _vwap_data()
	full = sc.rank_vwap_candidates(data, _codes(), None, criteria)
	top = sc.rank_vwap_candidates(data, _codes(), 10, criteria)
	assert full[:10] == top


# ==========
# Volume profile
# ==========
@pytest.mark.parametrize("criteria", VPROFILE_CRITERIA)
def test_vprofile_ranking_returns_every_candidate_when_unbounded(criteria):
	ranked = sc.rank_vprofile_candidates(_vprofile_frame(), criteria, None)
	assert len(ranked) == ROWS


@pytest.mark.parametrize("criteria", VPROFILE_CRITERIA)
def test_vprofile_ranking_still_truncates_for_the_legacy_default(criteria):
	assert len(sc.rank_vprofile_candidates(_vprofile_frame(), criteria, 10)) == 10


@pytest.mark.parametrize("criteria", VPROFILE_CRITERIA)
def test_vprofile_unbounded_starts_with_the_bounded_result(criteria):
	full = sc.rank_vprofile_candidates(_vprofile_frame(), criteria, None)
	top = sc.rank_vprofile_candidates(_vprofile_frame(), criteria, 10)
	assert full.index[:10].tolist() == top.index.tolist()


def test_an_invalid_criterion_still_raises_even_when_unbounded():
	with pytest.raises(ValueError):
		sc.rank_vprofile_candidates(_vprofile_frame(10), "not_a_criterion", None)
	with pytest.raises(ValueError):
		sc.rank_vwap_candidates(_vwap_data(10), _codes(10), None, "not_a_criterion")


# ==========
# No hidden cap
# ==========
def test_no_rank_helper_hardcodes_a_row_limit():
	# A literal cap, or a "large enough" default, would reintroduce the bug
	# quietly. The only truncation allowed is the caller's n_stockcodes.
	source = (pathlib.Path(sc.__file__)).read_text()
	for number in ("10", "50", "100", "500", "1000"):
		assert f".head({number})" not in source
		assert f".iloc[:{number}]" not in source


# ==========
# The fifth truncation point: foreign money flow truncates in SQL
# ==========
def test_a_null_limit_emits_no_limit_clause():
	# What the foreign money-flow path relies on. If SQLAlchemy ever rendered
	# `LIMIT NULL` or raised instead, that family would silently go back to
	# returning nothing or everything-but-wrong, and no pandas test would see it.
	bounded = str(select(db.StockData.code).limit(10).compile(compile_kwargs={"literal_binds": True}))
	unbounded = str(select(db.StockData.code).limit(None).compile(compile_kwargs={"literal_binds": True}))

	assert "LIMIT 10" in bounded
	assert "LIMIT" not in unbounded


def test_the_foreign_money_flow_query_limits_only_by_n_stockcodes():
	# The cap is passed straight through, so None reaches SQLAlchemy as None.
	# A literal here would be invisible to every helper test above.
	source = inspect.getsource(ff.ScreenerMoneyFlow._get_mf_top_stockcodes)

	assert source.count(".limit(n_stockcodes)") == 2
	for number in ("10", "50", "100", "500", "1000"):
		assert f".limit({number})" not in source
