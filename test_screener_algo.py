"""
Deterministic guards for the screener algorithms that were de-duplicated or
vectorised. No database: every frame here is synthetic, and each test compares
the shipped implementation against the straightforward reference it replaced.
"""
import asyncio
import datetime
import inspect

import numpy as np
import pandas as pd
import pytest
from fastapi.openapi.utils import get_openapi

import dependencies as dp
from quantist_library import brokerflow as bf
from quantist_library import foreignflow as ff
from quantist_library import screener as sc
from routers import whaleanalysis


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


def _vprofile_frame(code: str = "aaa", tail: tuple[float, ...] = ()) -> pd.DataFrame:
	close = [100.0, 104.0, 101.0, 108.0, 103.0, 112.0, 105.0, 118.0, 102.0, 109.0,
		101.0, 115.0, 103.0, 120.0, 106.0, 111.0, 100.0, 107.0, 104.0, 102.0] + list(tail)
	netval = [(-1.0) ** i * (i + 1) * 1000.0 for i in range(len(close))]
	dates = pd.bdate_range("2024-01-01", periods=len(close))
	return pd.DataFrame(
		{"close": np.array(close), "netval": np.array(netval)},
		index=pd.MultiIndex.from_product([[code], dates], names=["code", "date"]),
	)


def test_vprofile_inside_still_answers_membership():
	"""The zone annotations are additive: the old two-value contract is untouched."""
	code, inside = asyncio.run(sc.vprofile_inside(_vprofile_frame(), 3))
	assert code == "aaa"
	assert isinstance(inside, bool), "vprofile_inside answers one question: is price inside a zone"


# ==========
# Volume profile zone annotations
# ==========
_ZONES = pd.IntervalIndex.from_tuples([(100.0, 110.0), (130.0, 140.0)])


def _reading(closes: list[float], checking_period: int = 3, zones=_ZONES) -> dict:
	return sc.vprofile_reading(pd.Series(closes, dtype="float64"), zones, checking_period)


def test_vprofile_role_is_resistance_when_price_approaches_from_below():
	reading = _reading([90.0, 95.0, 105.0])
	assert reading["vprofile_zone_role"] == "resistance"
	assert reading["vprofile_in_zone"] is True


def test_vprofile_role_is_support_when_price_approaches_from_above():
	reading = _reading([125.0, 120.0, 105.0])
	assert reading["vprofile_zone_role"] == "support"


def test_vprofile_role_is_undetermined_without_outside_context():
	"""Every observed close sits in the zone: there is no approach to read."""
	reading = _reading([102.0, 104.0, 105.0])
	assert reading["vprofile_zone_role"] == "undetermined"


def test_vprofile_role_never_invents_a_zone_without_a_profile():
	reading = _reading([102.0, 104.0, 105.0], zones=pd.IntervalIndex.from_tuples([]))
	assert reading["vprofile_zone_role"] == "undetermined"
	assert reading["vprofile_zone_behavior"] == "undetermined"
	assert reading["vprofile_in_zone"] is False
	assert reading["vprofile_zone_mid"] is None


def test_vprofile_behavior_acceptance_needs_consecutive_closes_inside():
	assert _reading([90.0, 104.0, 105.0])["vprofile_zone_behavior"] == "acceptance"


def test_vprofile_behavior_test_is_one_isolated_touch():
	assert _reading([90.0, 95.0, 105.0])["vprofile_zone_behavior"] == "test"


@pytest.mark.parametrize("closes, role, behavior", [
	# came up into the zone, fell back out below it: rejected at resistance
	([90.0, 105.0, 95.0], "resistance", "rejection"),
	# came down onto the zone, bounced back above it: held as support
	([125.0, 105.0, 115.0], "support", "rejection"),
	# came up into the zone and left above it: resistance broken
	([90.0, 105.0, 115.0], "resistance", "breakout_up"),
	# came down onto the zone and left below it: support broken
	([125.0, 105.0, 95.0], "support", "breakdown"),
])
def test_vprofile_behavior_reads_the_exit_against_the_approach(closes, role, behavior):
	reading = _reading(closes)
	assert (reading["vprofile_zone_role"], reading["vprofile_zone_behavior"]) == (role, behavior)
	assert reading["vprofile_in_zone"] is True, "the window still touched the zone"


def test_vprofile_behavior_is_undetermined_when_the_window_never_touched():
	reading = _reading([90.0, 92.0, 95.0])
	assert reading["vprofile_in_zone"] is False
	assert reading["vprofile_zone_behavior"] == "undetermined"
	assert reading["vprofile_touch_count"] == 0
	# The nearest zone is still reported so the distance can be read off it.
	assert reading["vprofile_zone_mid"] == pytest.approx(105.0)
	assert reading["vprofile_distance_to_mid_pct"] < 0


def test_vprofile_reading_reports_the_zone_levels_and_the_distance():
	reading = _reading([90.0, 95.0, 105.0])
	assert (reading["vprofile_zone_low"], reading["vprofile_zone_high"]) == (100.0, 110.0)
	assert reading["vprofile_zone_mid"] == pytest.approx(105.0)
	assert reading["vprofile_distance_to_mid_pct"] == pytest.approx(0.0)
	assert reading["vprofile_touch_count"] == 1


def test_vprofile_reading_selects_the_zone_the_price_is_working_on():
	reading = _reading([105.0, 120.0, 135.0])
	assert (reading["vprofile_zone_low"], reading["vprofile_zone_high"]) == (130.0, 140.0)


def test_vprofile_reading_is_json_safe():
	reading = _reading([90.0, 95.0, 105.0])
	for name, value in reading.items():
		assert type(value) in (bool, str, float, int, type(None)), f"{name} is {type(value)}"
	assert reading["vprofile_zone_role"] in sc.VPROFILE_ROLES
	assert reading["vprofile_zone_behavior"] in sc.VPROFILE_BEHAVIORS


def test_vprofile_reading_says_nothing_about_now_when_the_last_close_is_missing():
	reading = _reading([90.0, 105.0, float("nan")])
	assert reading["vprofile_zone_role"] == "undetermined"
	assert reading["vprofile_zone_behavior"] == "undetermined"
	assert reading["vprofile_distance_to_mid_pct"] is None


def test_vprofile_annotate_agrees_with_the_membership_it_annotates():
	data = _vprofile_frame()
	code, reading = asyncio.run(sc.vprofile_annotate(data, 3))
	assert code == "aaa"
	assert reading["vprofile_in_zone"] == asyncio.run(sc.vprofile_inside(data, 3))[1]
	assert reading["vprofile_zone_role"] in sc.VPROFILE_ROLES
	assert reading["vprofile_zone_behavior"] in sc.VPROFILE_BEHAVIORS


def test_vprofile_annotations_frame_is_indexed_by_code():
	data = pd.concat([_vprofile_frame("aaa"), _vprofile_frame("bbb")])
	frame = asyncio.run(sc.vprofile_annotations(data, 3))
	assert frame.index.tolist() == ["aaa", "bbb"]
	assert "vprofile_zone_role" in frame.columns
	# The serialisation the router hands to orjson keeps native scalars.
	for values in frame.to_dict(orient="index").values():
		for value in values.values():
			assert type(value) in (bool, str, float, int, type(None))


def _annotation_columns() -> set[str]:
	return set(_reading([90.0, 95.0, 105.0]))


def test_foreign_vprofile_screener_annotates_its_top_stockcodes():
	screener = object.__new__(ff.ScreenerVProfile)
	screener.radar_period = 3
	screener.raw_data = pd.concat([_vprofile_frame("aaa"), _vprofile_frame("bbb")])
	screener.stocklist = ["aaa", "bbb"]
	screener.close_valflow_corr = pd.Series({"aaa": 0.4, "bbb": 0.6})

	stocklist, top = asyncio.run(screener._get_data_from_stocklist(n_stockcodes=2))

	assert set(stocklist) == {"aaa", "bbb"}
	# The columns the API already serves stay first-class...
	assert {"close", "mf", "corr"} <= set(top.columns)
	# ...and the annotations are additive.
	assert _annotation_columns() <= set(top.columns)


def test_broker_vprofile_screener_annotates_its_top_stockcodes():
	data = pd.concat([_vprofile_frame("aaa"), _vprofile_frame("bbb")])
	screener = object.__new__(bf.ScreenerVProfile)
	screener.radar_period = 3
	screener.raw_data_full = data
	screener.wf_indicators = data
	screener.selected_broker_nval = data[["netval"]].rename(columns={"netval": "broker_nval"})
	screener.stocklist = ["aaa", "bbb"]
	screener.optimum_corr = pd.Series({"aaa": 0.4, "bbb": 0.6})

	stocklist, top = asyncio.run(screener._get_data_from_stocklist(n_stockcodes=2))

	assert set(stocklist) == {"aaa", "bbb"}
	assert {"close", "mf", "corr"} <= set(top.columns)
	assert _annotation_columns() <= set(top.columns)


# ==========
# Volume profile behavior criteria: breakout / cross down
# ==========
def _behavior_annotations() -> pd.DataFrame:
	"""One hand-written reading per behavior the criteria have to tell apart."""
	rows = {
		"brk": ("resistance", "breakout_up"),
		"dwn": ("support", "breakdown"),
		"acc": ("resistance", "acceptance"),
		"tst": ("support", "test"),
		"rej": ("support", "rejection"),		# role, but price never left the zone
		"und": ("resistance", "undetermined"),	# role, nothing else observed
	}
	return pd.DataFrame(
		[{"vprofile_in_zone": True, "vprofile_zone_role": role, "vprofile_zone_behavior": behavior}
			for role, behavior in rows.values()],
		index=pd.Index(rows, name="code"),
	)


@pytest.mark.parametrize("behavior, expected", [("breakout_up", ["brk"]), ("breakdown", ["dwn"])])
def test_vprofile_behavior_codes_selects_only_the_named_behavior(behavior, expected):
	selected = sc.vprofile_behavior_codes(_behavior_annotations(), behavior)
	assert selected == expected
	# Sitting in a zone is not a signal, and neither is carrying a role on its own.
	assert not {"acc", "tst", "rej", "und"} & set(selected)


def _behavior_universe() -> pd.DataFrame:
	"""Four codes on the same profile, each leaving (or not leaving) its zone differently."""
	tails = {
		"brk": (108.0, 118.0, 130.0),	# came up into the zone and left above it
		"dwn": (100.0, 88.0, 80.0),		# came down onto the zone and left below it
		"acc": (),						# still sitting inside its zone
		"rej": (104.0, 125.0, 140.0),	# touched and turned back: role only, no break
	}
	return pd.concat([_vprofile_frame(code, tail) for code, tail in tails.items()])


def test_vprofile_criteria_read_the_behavior_not_the_membership():
	data = _behavior_universe()
	annotations = asyncio.run(sc.vprofile_annotations(data, 3))
	breakout = asyncio.run(sc.vprofile_breakout(data, 3))
	cross_down = asyncio.run(sc.vprofile_cross_down(data, 3))

	assert breakout == ["brk"]
	assert cross_down == ["dwn"]
	assert annotations.loc[breakout + cross_down, "vprofile_zone_behavior"].tolist() == ["breakout_up", "breakdown"]
	# All four touched a zone: membership is the wider set the signals are read out of.
	assert set(asyncio.run(sc.vprofile_stocklist(data, 3))) == {"brk", "dwn", "acc", "rej"}


@pytest.mark.parametrize("criteria, expected", [
	(dp.ScreenerList.vprofile_inside, ["acc", "brk", "dwn", "rej"]),
	(dp.ScreenerList.vprofile_breakout, ["brk"]),
	(dp.ScreenerList.vprofile_cross_down, ["dwn"]),
])
def test_vprofile_criteria_stocklist_dispatches_on_the_enum(criteria, expected):
	assert sorted(asyncio.run(sc.vprofile_criteria_stocklist(_behavior_universe(), 3, criteria))) == expected


def test_vprofile_criteria_stocklist_rejects_an_unknown_criterion():
	with pytest.raises(ValueError):
		asyncio.run(sc.vprofile_criteria_stocklist(_behavior_universe(), 3, dp.ScreenerList.vwap_rally))


@pytest.mark.parametrize("criteria, expected", [
	(dp.ScreenerList.vprofile_inside, ["acc", "brk", "dwn", "rej"]),
	(dp.ScreenerList.vprofile_breakout, ["brk"]),
	(dp.ScreenerList.vprofile_cross_down, ["dwn"]),
])
def test_foreign_vprofile_screener_selects_on_the_requested_criterion(criteria, expected):
	screener = object.__new__(ff.ScreenerVProfile)
	screener.radar_period = 3
	screener.screener_vprofile_criteria = criteria
	stocklist = asyncio.run(screener._get_vprofile_stocklist(raw_data=_behavior_universe()))
	assert sorted(stocklist) == expected


@pytest.mark.parametrize("criteria, expected", [
	(dp.ScreenerList.vprofile_inside, ["acc", "brk", "dwn", "rej"]),
	(dp.ScreenerList.vprofile_breakout, ["brk"]),
	(dp.ScreenerList.vprofile_cross_down, ["dwn"]),
])
def test_broker_vprofile_screener_selects_on_the_requested_criterion(criteria, expected):
	screener = object.__new__(bf.ScreenerVProfile)
	screener.radar_period = 3
	screener.screener_vprofile_criteria = criteria
	screener.wf_indicators = _behavior_universe()
	assert sorted(asyncio.run(screener._get_vprofile_stocklist())) == expected


@pytest.mark.parametrize("cls", [ff.ScreenerVProfile, bf.ScreenerVProfile])
def test_vprofile_screener_constructors_default_to_membership(cls):
	parameter = inspect.signature(cls.__init__).parameters["screener_vprofile_criteria"]
	assert parameter.default == dp.ScreenerList.vprofile_inside


@pytest.mark.parametrize("path", [
	"/whaleanalysis/screener/foreign/vprofile",
	"/whaleanalysis/screener/broker/vprofile",
])
def test_vprofile_routes_expose_the_criterion_as_an_optional_query_parameter(path):
	"""OpenAPI only: no server, no database, just the route signatures FastAPI reads."""
	schema = get_openapi(title="test", version="test", routes=whaleanalysis.router.routes)
	parameters = {p["name"]: p for p in schema["paths"][path]["get"]["parameters"]}
	criterion = parameters["screener_vprofile_criteria"]
	assert criterion["in"] == "query"
	assert criterion["required"] is False
	assert criterion["schema"]["default"] == "vprofile_inside"
	assert set(criterion["schema"]["enum"]) == {"vprofile_inside", "vprofile_breakout", "vprofile_cross_down"}


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
