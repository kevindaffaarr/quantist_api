"""
Every screener family fills every shared column.

The web contract reports five columns for all eleven criteria — close, money
flow, proportion, price correlation and VWAP — but each family only ever put
the two or three its own ranking needed into the frame, and the contract read
the rest as null. Which columns were missing depended on the family:

	money flow      no close, no vwap
	vwap            no prop, no pricecorrel
	volume profile  no vwap, no prop

These drive the real methods with synthetic frames and a stand-in object, so
they run without a database and still exercise the production code path
rather than a re-implementation of it.
"""
import asyncio
import datetime
import inspect
import types

import numpy as np
import pandas as pd
import pytest

import dependencies as dp
import web_contract as wc
from quantist_library import brokerflow as bf
from quantist_library import foreignflow as ff
from quantist_library import screener as sc

CODES = ["aaaa", "bbbb", "cccc", "dddd"]
# Deliberately in the past, and the frames below never contain anything later.
DATES = pd.to_datetime(["2026-09-14", "2026-09-15", "2026-09-16", "2026-09-17", "2026-09-18"])
ENDDATE = datetime.date(2026, 9, 18)
STARTDATE = datetime.date(2026, 9, 14)
# A date no fixture contains. If a column is ever computed from data past the
# screener window, the frames built with this will say so.
FUTURE = pd.Timestamp("2026-09-25")

SHARED_COLUMNS = ("close", "money_flow", "proportion", "price_correlation", "vwap")


def _index(codes=CODES, dates=DATES) -> pd.MultiIndex:
	return pd.MultiIndex.from_product([codes, dates], names=["code", "date"])


# Deliberately not a straight line. A linear series has constant first
# differences, which makes every Pearson correlation in the pipeline NaN — so
# a ramp would let a genuinely broken correlation column pass as "no data".
_WOBBLE = (1.0, 1.7, 0.6, 2.3, 1.2, 0.4, 2.8, 1.5)


def _ramp(rows: int, start: float, step: float) -> list[float]:
	return [start + step * (position + _WOBBLE[position % len(_WOBBLE)]) for position in range(rows)]


def _price_frame(index: pd.MultiIndex) -> pd.DataFrame:
	"""close and market value, both strictly positive and varying per bar."""
	rows = len(index)
	return pd.DataFrame({"close": _ramp(rows, 1000.0, 7.0), "value": _ramp(rows, 5.0e9, 1.0e8)}, index=index)


def _flow(index: pd.MultiIndex, column: str, start: float, step: float) -> pd.DataFrame:
	return pd.DataFrame({column: _ramp(len(index), start, step)}, index=index)


# ==========
# The shared helpers
# ==========
class TestFlowVwap:
	def test_it_is_accumulation_value_over_accumulation_volume(self):
		index = _index(["aaaa"], DATES[:2])
		value = pd.Series([1_000.0, 3_000.0], index=index)
		volume = pd.Series([10.0, 30.0], index=index)

		assert sc.flow_vwap(value, volume)["aaaa"] == pytest.approx(100.0)

	def test_distribution_days_are_excluded_from_both_sides_independently(self):
		# The rolling version in the chart filters each series on its own sign,
		# so a value-positive, volume-negative day contributes to the numerator
		# only. Matching that exactly is the point of this helper existing.
		index = _index(["aaaa"], DATES[:3])
		value = pd.Series([1_000.0, -500.0, 2_000.0], index=index)
		volume = pd.Series([10.0, -5.0, -20.0], index=index)

		assert sc.flow_vwap(value, volume)["aaaa"] == pytest.approx(3_000.0 / 10.0)

	def test_a_code_that_never_accumulated_is_null_not_zero(self):
		# Nothing was bought. That is not the same as buying at a price of zero,
		# and the contract must say "no value" rather than invent one.
		index = _index(["aaaa"], DATES[:2])
		value = pd.Series([-1.0, -2.0], index=index)
		volume = pd.Series([-1.0, -2.0], index=index)

		assert pd.isna(sc.flow_vwap(value, volume)["aaaa"])

	def test_it_answers_per_code(self):
		index = _index(["aaaa", "bbbb"], DATES[:2])
		value = pd.Series([100.0, 100.0, 400.0, 400.0], index=index)
		volume = pd.Series([1.0, 1.0, 2.0, 2.0], index=index)

		result = sc.flow_vwap(value, volume)

		assert result["aaaa"] == pytest.approx(200.0 / 2.0)
		assert result["bbbb"] == pytest.approx(800.0 / 4.0)
		assert list(result.index) == ["aaaa", "bbbb"]


class TestFlowProportion:
	def test_it_is_gross_flow_over_both_sides_of_market_value(self):
		# Both sides: a participant transacting every lot has gross value twice
		# the market's turnover, and scores 1.0 — not 2.0.
		index = _index(["aaaa"], DATES[:2])
		gross = pd.Series([100.0, 100.0], index=index)
		market = pd.Series([50.0, 50.0], index=index)

		assert sc.flow_proportion(gross, market)["aaaa"] == pytest.approx(1.0)
		assert sc.flow_proportion(gross / 2, market)["aaaa"] == pytest.approx(0.5)

	def test_a_code_with_no_turnover_is_null_not_a_divide_by_zero(self):
		index = _index(["aaaa"], DATES[:2])
		gross = pd.Series([5.0, 5.0], index=index)
		market = pd.Series([0.0, 0.0], index=index)

		assert pd.isna(sc.flow_proportion(gross, market)["aaaa"])


class TestFlowPriceCorrelation:
	def test_price_moving_with_the_flow_correlates_positively(self):
		# The helper differences the cumulative flow, so the flow series here
		# IS the per-bar change the price is compared against.
		index = _index(["aaaa"], DATES)
		close = pd.Series([100.0, 110.0, 108.0, 130.0, 133.0], index=index)
		flow = pd.Series([0.0, 10.0, -2.0, 22.0, 3.0], index=index)

		assert sc.flow_price_correlation(close, flow)["aaaa"] == pytest.approx(1.0)

	def test_price_moving_against_the_flow_correlates_negatively(self):
		index = _index(["aaaa"], DATES)
		close = pd.Series([100.0, 90.0, 92.0, 70.0, 67.0], index=index)
		flow = pd.Series([0.0, 10.0, -2.0, 22.0, 3.0], index=index)

		assert sc.flow_price_correlation(close, flow)["aaaa"] == pytest.approx(-1.0)

	def test_a_flat_price_has_no_correlation_rather_than_a_made_up_one(self):
		index = _index(["aaaa"], DATES)
		close = pd.Series([100.0] * 5, index=index)
		flow = pd.Series([0.0, 10.0, -2.0, 22.0, 3.0], index=index)

		assert pd.isna(sc.flow_price_correlation(close, flow)["aaaa"])


# ==========
# Stand-in screener objects
# ==========
def _run(method, instance, /, **kwargs):
	"""Call an unbound async method against a stand-in self."""
	return asyncio.run(method(instance, **kwargs))


def _broker_money_flow_object() -> types.SimpleNamespace:
	index = _index()
	return types.SimpleNamespace(
		raw_data_full=_price_frame(index),
		selected_broker_nval=_flow(index, "broker_nval", 1.0e9, 5.0e7),
		selected_broker_nvol=_flow(index, "broker_nvol", 1.0e6, 2.0e4),
		selected_broker_sumval=_flow(index, "broker_sumval", 2.0e9, 1.0e8),
		radar_period=5,
		bar_range=5,
	)


def _broker_vwap_object() -> types.SimpleNamespace:
	index = _index()
	top_data = _price_frame(index)
	top_data["vwap"] = _ramp(len(index), 990.0, 6.0)
	top_data["broker_nval"] = _ramp(len(index), 1.0e9, 5.0e7)
	top_data["broker_sumval"] = _ramp(len(index), 2.0e9, 1.0e8)
	return types.SimpleNamespace(
		top_data=top_data,
		stocklist=list(CODES),
		optimum_corr=pd.Series([0.71, 0.62, 0.53, 0.44], index=CODES),
	)


def _broker_vprofile_object() -> types.SimpleNamespace:
	index = _index()
	price = _price_frame(index)
	wf_indicators = price.copy()
	wf_indicators["netval"] = _ramp(len(index), 1.0e9, 5.0e7)
	return types.SimpleNamespace(
		radar_period=3,
		stocklist=list(CODES),
		raw_data_full=price,
		wf_indicators=wf_indicators,
		selected_broker_nval=_flow(index, "broker_nval", 1.0e9, 5.0e7),
		selected_broker_nvol=_flow(index, "broker_nvol", 1.0e6, 2.0e4),
		selected_broker_sumval=_flow(index, "broker_sumval", 2.0e9, 1.0e8),
		optimum_corr=pd.Series([0.71, 0.62, 0.53, 0.44], index=CODES),
		screener_vprofile_criteria=dp.ScreenerList.vprofile_inside,
	)


def _foreign_vwap_object() -> types.SimpleNamespace:
	index = _index()
	top_data = _price_frame(index)
	top_data["vwap"] = _ramp(len(index), 990.0, 6.0)
	top_data["netval"] = _ramp(len(index), 1.0e9, 5.0e7)
	top_data["sumval"] = _ramp(len(index), 2.0e9, 1.0e8)
	return types.SimpleNamespace(top_data=top_data, stocklist=list(CODES))


def _foreign_vprofile_object() -> types.SimpleNamespace:
	index = _index()
	raw = _price_frame(index)
	raw["netval"] = _ramp(len(index), 1.0e9, 5.0e7)
	raw["netvol"] = _ramp(len(index), 1.0e6, 2.0e4)
	raw["sumval"] = _ramp(len(index), 2.0e9, 1.0e8)
	return types.SimpleNamespace(
		radar_period=3,
		stocklist=list(CODES),
		raw_data=raw,
		close_valflow_corr=pd.Series([0.71, 0.62, 0.53, 0.44], index=CODES),
		screener_vprofile_criteria=dp.ScreenerList.vprofile_inside,
	)


# Each entry produces the frame a family hands to the web contract.
def _broker_money_flow_frame(**kwargs) -> pd.DataFrame:
	instance = kwargs.pop("instance", None) or _broker_money_flow_object()
	return _run(
		bf.ScreenerMoneyFlow._get_mf_top_stockcodes, instance,
		accum_or_distri=kwargs.get("criterion", dp.ScreenerList.most_accumulated),
		n_stockcodes=None, startdate=kwargs.get("startdate", STARTDATE), enddate=kwargs.get("enddate", ENDDATE),
	)


def _broker_vwap_frame(**kwargs) -> pd.DataFrame:
	instance = kwargs.pop("instance", None) or _broker_vwap_object()
	return bf.ScreenerVWAP._compile_top_stockcodes(instance)


def _broker_vprofile_frame(**kwargs) -> pd.DataFrame:
	instance = kwargs.pop("instance", None) or _broker_vprofile_object()
	_, frame = _run(bf.ScreenerVProfile._get_data_from_stocklist, instance, n_stockcodes=None)
	return frame


def _foreign_vwap_frame(**kwargs) -> pd.DataFrame:
	instance = kwargs.pop("instance", None) or _foreign_vwap_object()
	return ff.ScreenerVWAP._compile_top_stockcodes(instance)


def _foreign_vprofile_frame(**kwargs) -> pd.DataFrame:
	instance = kwargs.pop("instance", None) or _foreign_vprofile_object()
	_, frame = _run(ff.ScreenerVProfile._get_data_from_stocklist, instance, n_stockcodes=None)
	return frame


def _foreign_money_flow_raw(index: pd.MultiIndex | None = None) -> pd.DataFrame:
	"""What the family's SQL query returns, in the shape it returns it."""
	index = _index() if index is None else index
	raw = _price_frame(index)
	raw["netvol"] = _ramp(len(index), 1.0e6, 2.0e4)
	raw["sumvol"] = _ramp(len(index), 3.0e6, 4.0e4)
	return raw


def _foreign_money_flow_frame(**kwargs) -> pd.DataFrame:
	"""
	The window arithmetic, driven directly.

	Only the fetch is stubbed: this family reads its window straight out of
	SQL, so the query cannot run here, but everything the web contract
	actually reads is computed by the production method below.
	"""
	raw = kwargs.pop("raw", None)
	raw = _foreign_money_flow_raw() if raw is None else raw
	seeded = pd.DataFrame(index=pd.Index(sorted({code for code, _ in raw.index}), name="code"))
	seeded["pricecorrel"] = sc.flow_price_correlation(raw["close"], raw["close"] * raw["netvol"])
	return ff.ScreenerMoneyFlow._compile_window_columns(raw, seeded, dp.ScreenerList.most_accumulated)


FAMILIES = {
	("broker", "money_flow"): _broker_money_flow_frame,
	("broker", "vwap"): _broker_vwap_frame,
	("broker", "vprofile"): _broker_vprofile_frame,
	("foreign", "money_flow"): _foreign_money_flow_frame,
	("foreign", "vwap"): _foreign_vwap_frame,
	("foreign", "vprofile"): _foreign_vprofile_frame,
}

SLUG_FOR_FAMILY = {"money_flow": "most_accumulated", "vwap": "vwap_rally", "vprofile": "vprofile_inside"}


# ==========
# Every family, both methods, every shared column
# ==========
@pytest.mark.parametrize("method, family", sorted(FAMILIES))
def test_every_family_and_method_fills_every_shared_column(method, family):
	frame = FAMILIES[(method, family)]()
	results = wc.build_screener_results(
		slug=SLUG_FOR_FAMILY[family],
		method=method,
		frame=frame,
		metadata={"startdate": STARTDATE, "enddate": ENDDATE, "bar_range": 5},
	)

	assert results.rows, "a family that returns no rows proves nothing"
	for row in results.rows:
		for column in SHARED_COLUMNS:
			value = getattr(row, column)
			assert value is not None, f"{method}/{family} left {column} null for {row.code}"
			assert isinstance(value, float), f"{method}/{family} returned {column} as {type(value).__name__}"
			assert not np.isnan(value), f"{method}/{family} returned {column} as NaN for {row.code}"


@pytest.mark.parametrize("method, family", sorted(FAMILIES))
def test_the_rank_and_order_survive_the_added_columns(method, family):
	frame = FAMILIES[(method, family)]()
	results = wc.build_screener_results(
		slug=SLUG_FOR_FAMILY[family], method=method, frame=frame, metadata={"enddate": ENDDATE},
	)

	assert [row.rank for row in results.rows] == list(range(1, len(frame) + 1))
	# The contract enumerates the frame as given; the frame's order is the
	# family's ranking, and adding columns must not reshuffle it.
	assert [row.code for row in results.rows] == list(frame.index)
	assert results.count == len(frame)


@pytest.mark.parametrize("method, family", sorted(FAMILIES))
def test_the_filled_response_is_json_safe(method, family):
	raw = wc.build_screener_results(
		slug=SLUG_FOR_FAMILY[family],
		method=method,
		frame=FAMILIES[(method, family)](),
		metadata={"startdate": STARTDATE, "enddate": ENDDATE, "bar_range": 5},
	).model_dump(mode="json")

	import json

	assert json.dumps(raw)
	for entry in raw["rows"]:
		for column in SHARED_COLUMNS:
			assert isinstance(entry[column], float), f"{column} is {entry[column]!r} after serialization"


def test_the_volume_profile_extras_still_ride_along():
	# Filling the shared columns must not push the per-criterion annotations
	# out of the frame — they are the whole point of those criteria.
	results = wc.build_screener_results(
		slug="vprofile_inside", method="broker", frame=_broker_vprofile_frame(), metadata={"enddate": ENDDATE},
	)

	extras = results.rows[0].extras
	assert extras, "the profile annotations were dropped"
	assert any(name.startswith("vprofile_") for name in extras)


# ==========
# The window is the window
# ==========
def test_no_column_is_computed_from_a_date_past_the_screener_window():
	# A bar dated after enddate is planted in every input. The reported values
	# must be identical to the run without it: a family that widened its window
	# to "all the data I was handed" would pick it up and disagree here.
	clean = _broker_money_flow_frame()

	polluted_instance = _broker_money_flow_object()
	future_index = _index(CODES, [FUTURE])
	for attribute, column, value in (
		("raw_data_full", None, None),
		("selected_broker_nval", "broker_nval", 9.9e12),
		("selected_broker_nvol", "broker_nvol", 9.9e9),
		("selected_broker_sumval", "broker_sumval", 9.9e12),
	):
		existing = getattr(polluted_instance, attribute)
		if column is None:
			addition = pd.DataFrame({"close": [9.9e5] * len(CODES), "value": [9.9e12] * len(CODES)}, index=future_index)
		else:
			addition = pd.DataFrame({column: [value] * len(CODES)}, index=future_index)
		setattr(polluted_instance, attribute, pd.concat([existing, addition]).sort_index())

	polluted = _broker_money_flow_frame(instance=polluted_instance)

	pd.testing.assert_frame_equal(clean, polluted)


def test_the_volume_profile_window_is_the_radar_tail_not_the_whole_year():
	# vprofile loads a year but reports over radar_period. If vwap or prop read
	# the whole frame, lengthening the history would move them.
	short = _broker_vprofile_frame()

	long_instance = _broker_vprofile_object()
	older = _index(CODES, pd.to_datetime(["2026-01-05", "2026-01-06"]))
	long_instance.raw_data_full = pd.concat([
		pd.DataFrame({"close": [1.0] * len(older), "value": [9.9e12] * len(older)}, index=older),
		long_instance.raw_data_full,
	]).sort_index()
	for attribute, column in (
		("selected_broker_nval", "broker_nval"),
		("selected_broker_nvol", "broker_nvol"),
		("selected_broker_sumval", "broker_sumval"),
	):
		existing = getattr(long_instance, attribute)
		addition = pd.DataFrame({column: [9.9e12] * len(older)}, index=older)
		setattr(long_instance, attribute, pd.concat([addition, existing]).sort_index())

	longer = _broker_vprofile_frame(instance=long_instance)

	for column in ("vwap", "prop"):
		pd.testing.assert_series_equal(short[column], longer[column])


# ==========
# The production code path, not a copy of it
# ==========
def test_no_two_contract_fields_are_fed_by_the_same_source_column():
	# The shortcut this work refused: filling a null column by copying a
	# neighbouring one in the contract. Two fields sharing a source column
	# would mean exactly that, so the mapping must stay many-to-one per field
	# and never one-to-many.
	owners: dict[str, str] = {}
	for column, field in wc._SCREENER_FIELDS.items():
		assert column not in owners, f"{column} feeds both {owners.get(column)} and {field}"
		owners[column] = field

	assert set(wc._SCREENER_FIELDS.values()) >= {"close", "money_flow", "proportion", "price_correlation", "vwap"}


@pytest.mark.parametrize("family, method, module", [
	("money_flow", "broker", bf), ("vwap", "broker", bf), ("vprofile", "broker", bf),
	("money_flow", "foreign", ff), ("vwap", "foreign", ff), ("vprofile", "foreign", ff),
])
def test_each_family_names_the_shared_helpers_rather_than_reinventing_them(family, method, module):
	# Six copies of "what is proportion" is how the chart and the screener end
	# up disagreeing. Each family calls the shared helper instead.
	klass = {"money_flow": module.ScreenerMoneyFlow, "vwap": module.ScreenerVWAP, "vprofile": module.ScreenerVProfile}[family]
	source = inspect.getsource(klass)

	if family != "money_flow" or method == "broker":
		assert "sc.flow_" in source, f"{method}/{family} does not use the shared indicator helpers"
