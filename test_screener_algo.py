"""
Deterministic guards for the screener algorithms that were de-duplicated or
vectorised. No database: every frame here is synthetic, and each test compares
the shipped implementation against the straightforward reference it replaced.
"""
import datetime

import numpy as np
import pandas as pd
import pytest

from quantist_library import brokerflow as bf
from quantist_library import foreignflow as ff
from quantist_library import screener as sc


def _vwap_frame() -> pd.DataFrame:
	"""(code, date) x [close, vwap] covering rally / around / breakout / breakdown."""
	dates = pd.to_datetime([datetime.date(2024, 1, d) for d in (2, 3, 4, 5, 8)])
	rows = {
		# always above -> rally, and last close is above -> breakout candidate
		"aaa": ([110, 111, 112, 113, 114], [100, 100, 100, 100, 100]),
		# crosses up on the 4th bar and stays above -> breakout
		"bbb": ([90, 95, 99, 101, 110], [100, 100, 100, 100, 100]),
		# crosses down on the 4th bar and stays below -> breakdown
		"ccc": ([110, 105, 101, 99, 90], [100, 100, 100, 100, 100]),
		# hugs vwap -> around
		"ddd": ([101, 99, 100, 102, 98], [100, 100, 100, 100, 100]),
		# always below -> nothing but breakdown-candidate without a cross
		"eee": ([80, 79, 78, 77, 76], [100, 100, 100, 100, 100]),
	}
	frames = []
	for code, (close, vwap) in rows.items():
		frames.append(pd.DataFrame(
			{"close": np.array(close, dtype=float), "vwap": np.array(vwap, dtype=float)},
			index=pd.MultiIndex.from_product([[code], dates], names=["code", "date"]),
		))
	return pd.concat(frames)


def _reference_cross(data: pd.DataFrame, above: bool) -> list:
	"""The rolling(2).apply() formulation the vectorised criteria replaced."""
	last = data[["close", "vwap"]].groupby(level="code").last()
	if above:
		stocklist = last[last["close"] >= last["vwap"]].index.tolist()
	else:
		stocklist = last[last["close"] <= last["vwap"]].index.tolist()
	top = data.loc[data.index.get_level_values("code").isin(stocklist)].copy()
	top["flag"] = top["close"] >= top["vwap"] if above else top["close"] <= top["vwap"]
	top["cross"] = top.groupby(level="code").rolling(window=2)["flag"]\
		.apply(lambda x: (x.iloc[0] == 0) & (x.iloc[1] == 1)).droplevel(0)
	hit = top["cross"].groupby(level="code").any()
	return hit[hit].index.tolist()


def test_vwap_rally_and_around():
	data = _vwap_frame()
	assert sc.vwap_rally(data) == ["aaa"]
	assert sc.vwap_around(data, percentage_range=0.05) == ["ddd"]


@pytest.mark.parametrize("above", [True, False])
def test_vwap_cross_matches_rolling_reference(above):
	data = _vwap_frame()
	got = sc.vwap_breakout(data) if above else sc.vwap_breakdown(data)
	assert got == _reference_cross(data, above=above)


@pytest.mark.parametrize("criteria", ["rally", "around", "breakout", "breakdown"])
def test_both_families_agree_on_vwap_criteria(criteria):
	"""Foreign and Whale read different value columns but must screen identically."""
	data = _vwap_frame()
	foreign = ff.ScreenerVWAP.__new__(ff.ScreenerVWAP)
	whale = bf.ScreenerVWAP.__new__(bf.ScreenerVWAP)
	if criteria == "around":
		assert foreign._get_vwap_around(data, 0.05) == whale._get_vwap_around(data, 0.05)
	else:
		fn = f"_get_vwap_{criteria}"
		assert getattr(foreign, fn)(data) == getattr(whale, fn)(data)


def _broker_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
	"""(code, date) x broker net values, plus per-(code, broker) cluster correlations."""
	dates = pd.to_datetime([datetime.date(2024, 1, d) for d in (2, 3, 4)])
	index = pd.MultiIndex.from_product([["aaa", "bbb"], dates], names=["code", "date"])
	df = pd.DataFrame(
		np.arange(1.0, 19.0).reshape(6, 3),
		index=index,
		columns=pd.Index(["ax", "bx", "cx"], name="broker"),
	)
	features = pd.DataFrame(
		{"corr_cluster": [0.8, -0.4, 0.1, -0.9, 0.5, -0.2]},
		index=pd.MultiIndex.from_product([["aaa", "bbb"], ["ax", "bx", "cx"]], names=["code", "broker"]),
	)
	return df, features


def _reference_plusmin_by_code(df: pd.DataFrame, broker_cluster: pd.DataFrame) -> pd.DataFrame:
	"""The per-code groupby().apply() the vectorised sign flip replaced."""
	return df.groupby(level="code", group_keys=False).apply(
		lambda x: pd.concat([
			x.loc[:, broker_cluster.loc[broker_cluster["corr_cluster"] < 0].loc[x.name].index
				.get_level_values("broker").tolist()].mul(-1, axis=1),
			x.loc[:, broker_cluster.loc[broker_cluster["corr_cluster"] >= 0].loc[x.name].index
				.get_level_values("broker").tolist()],
		], axis=1)
	).sort_index(axis=1)


def test_adjust_plusmin_by_code_matches_groupby_reference():
	df, features = _broker_frame()
	pd.testing.assert_frame_equal(
		bf.adjust_plusmin_by_code(df, features),
		_reference_plusmin_by_code(df, features),
	)


def test_adjust_plusmin_single_code_flips_only_negative_brokers():
	df, features = _broker_frame()
	one = df.loc["aaa"]
	got = bf.adjust_plusmin(one, features.loc["aaa"])
	pd.testing.assert_series_equal(got["ax"], one["ax"])
	pd.testing.assert_series_equal(got["bx"], -one["bx"], check_names=False)
	pd.testing.assert_series_equal(got["cx"], one["cx"])


def _long_broker_rows() -> pd.DataFrame:
	"""Long-format broker rows, deliberately sparse so zero-fill matters."""
	rows = []
	for code in ("aaa", "bbb"):
		for day in (2, 3, 4, 5):
			for broker in ("ax", "bx", "cx"):
				if code == "bbb" and broker == "cx" and day < 5:
					continue  # broker only shows up on the last bar
				rows.append({
					"code": code,
					"date": pd.Timestamp(datetime.date(2024, 1, day)),
					"broker": broker,
					"nval": float(day * 10 + len(broker) + ord(broker[0])),
				})
	return pd.DataFrame(rows).set_index(["code", "date"])


def test_tail_pivot_aligns_to_the_full_period_pivot():
	"""Fetching only the tail window must reproduce the rows a full pivot would tail."""
	rows = _long_broker_rows()
	full = bf.pivot_broker_values(rows, "nval")
	expected = full.groupby(level="code").tail(2)

	cutoff = expected.index.get_level_values("date").min()
	tail_rows = rows[rows.index.get_level_values("date") >= cutoff]
	got = bf.pivot_broker_values(tail_rows, "nval", index=expected.index, columns=full.columns)

	pd.testing.assert_frame_equal(got, expected)


def test_pivot_broker_values_zero_fills_absent_brokers():
	full = bf.pivot_broker_values(_long_broker_rows(), "nval")
	assert list(full.columns) == ["ax", "bx", "cx"]
	assert full.loc[("bbb", pd.Timestamp("2024-01-02")), "cx"] == 0.0
