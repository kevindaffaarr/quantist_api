"""
Contract tests for the web chart payload.

The web frontend is the only consumer of web_contract, and it cannot repair a
malformed payload: NaN is not JSON, numpy scalars are not JSON, and a missing
profile boundary cannot be guessed from a mid point. So the normalizer is pure
and every guarantee the frontend relies on is pinned here with synthetic
frames, no database and no services.
"""
import datetime
import json

import numpy as np
import pandas as pd
import pytest

import web_contract as wc


def _indicators(rows: int = 4) -> pd.DataFrame:
	index = pd.to_datetime([datetime.date(2026, 9, 15) + datetime.timedelta(days=i) for i in range(rows)])
	return pd.DataFrame(
		{
			"openprice": np.arange(6400.0, 6400.0 + rows),
			"high": np.arange(6450.0, 6450.0 + rows),
			"low": np.arange(6350.0, 6350.0 + rows),
			"close": np.arange(6420.0, 6420.0 + rows),
			"value": np.full(rows, 9.5e12),
			"volume": np.full(rows, 1.8e10),
			"netval": np.linspace(-3e11, 1e11, rows),
			"valflow": np.linspace(-1e12, -2e11, rows),
			"mf": np.linspace(-1.5e12, -1.0e12, rows),
			"prop": np.linspace(0.30, 0.3342, rows),
			"netprop": np.linspace(0.10, 0.1221, rows),
			"pricecorrel": np.linspace(0.20, 0.4129, rows),
			"mapricecorrel": np.linspace(0.50, 0.6155, rows),
			"vwap": np.linspace(6500.0, 6569.35, rows),
			"pow": np.full(rows, 2),
		},
		index=index,
	)


def _holding() -> pd.DataFrame:
	index = pd.to_datetime([datetime.date(2026, 7, 1), datetime.date(2026, 8, 1)])
	return pd.DataFrame(
		{
			"foreign": [0.41, 0.40],
			"local_institutional": [0.38, 0.39],
			"local_individual": [0.21, 0.21],
			"scripless_ratio": [0.9712, 0.9720],
		},
		index=index,
	)


def _zones() -> pd.Series:
	bins = pd.IntervalIndex.from_breaks([6300.0, 6400.0, 6500.0, 6600.0])
	return pd.Series([-4.2e11, 8.1e11, -1.3e11], index=bins)


def _payload(**overrides):
	kwargs: dict = {
		"code": "composite",
		"wf_indicators": _indicators(),
		"analysis_method": "broker",
		"periods": {"mf": 1, "prop": 10, "pricecorrel": 10, "mapricecorrel": 100, "vwap": 21},
		"startdate": datetime.date(2026, 9, 15),
		"enddate": datetime.date(2026, 9, 18),
		"hist_bar": _zones(),
		"peaks_index": [1],
		"holding_composition": _holding(),
		"generated_at": datetime.datetime(2026, 9, 18, 17, 30, tzinfo=datetime.UTC),
	}
	kwargs.update(overrides)
	return wc.build_web_chart(**kwargs)


# ==========
# Instrument resolution / COMPOSITE alias
# ==========
def test_composite_is_the_default_instrument():
	assert wc.resolve_instrument(None).code == "composite"
	assert wc.resolve_instrument("").code == "composite"


@pytest.mark.parametrize("requested", ["composite", "COMPOSITE", "ihsg", "IHSG", " Ihsg "])
def test_ihsg_and_composite_resolve_to_the_same_index_instrument(requested):
	instrument = wc.resolve_instrument(requested)
	assert instrument.code == "composite"
	assert instrument.symbol == "COMPOSITE"
	assert instrument.display_name == "IHSG / COMPOSITE"
	assert instrument.kind == "index"
	assert "IHSG" in instrument.aliases


def test_a_plain_stock_code_is_not_aliased():
	instrument = wc.resolve_instrument("bbca")
	assert instrument.code == "bbca"
	assert instrument.symbol == "BBCA"
	assert instrument.display_name == "BBCA"
	assert instrument.kind == "stock"
	assert instrument.aliases == []


# ==========
# Sections
# ==========
def test_payload_has_every_semantic_section():
	payload = _payload()
	assert payload.schema_version == wc.SCHEMA_VERSION
	for section in ("instrument", "period", "summary", "price", "indicators", "profile_zones", "holding_composition", "annotations", "meta"):
		assert getattr(payload, section) is not None


def test_price_and_indicator_series_are_aligned_to_one_date_axis():
	payload = _payload()
	length = len(payload.price.dates)
	assert length == 4
	assert payload.period.bars == length
	for series in (payload.price.open, payload.price.high, payload.price.low, payload.price.close, payload.price.volume, payload.price.value):
		assert len(series) == length
	indicators = payload.indicators
	for series in (indicators.vwap, indicators.value_flow, indicators.proportion, indicators.net_proportion, indicators.net_value):
		assert len(series) == length


def test_summary_carries_the_latest_numeric_metrics_not_formatted_strings():
	summary = _payload().summary
	assert summary.date == datetime.date(2026, 9, 18)
	assert summary.close == pytest.approx(6423.0)
	assert summary.vwap == pytest.approx(6569.35)
	assert summary.money_flow == pytest.approx(-1.0e12)
	assert summary.proportion == pytest.approx(0.3342)
	assert summary.price_correlation == pytest.approx(0.4129)
	assert summary.ma_price_correlation == pytest.approx(0.6155)
	assert summary.method == "broker"
	assert summary.power == 2
	assert summary.power_label == "medium"
	for value in summary.model_dump().values():
		assert not isinstance(value, str) or value in {"broker", "medium"}


@pytest.mark.parametrize("pow_value, label", [(1, "low"), (2, "medium"), (3, "high")])
def test_power_label_follows_the_chart_thresholds(pow_value, label):
	frame = _indicators()
	frame["pow"] = pow_value
	assert _payload(wf_indicators=frame).summary.power_label == label


def test_profile_zones_expose_explicit_boundaries_and_peak_flags():
	zones = _payload().profile_zones
	assert [(z.low, z.high) for z in zones] == [(6300.0, 6400.0), (6400.0, 6500.0), (6500.0, 6600.0)]
	assert [z.mid for z in zones] == [6350.0, 6450.0, 6550.0]
	assert [z.is_peak for z in zones] == [False, True, False]
	assert zones[1].net_value == pytest.approx(8.1e11)


def test_holding_composition_keeps_the_three_chart_categories():
	holding = _payload().holding_composition
	assert holding is not None
	assert holding.periods == [datetime.date(2026, 7, 1), datetime.date(2026, 8, 1)]
	assert holding.foreign == pytest.approx([0.41, 0.40])
	assert holding.local_institutional == pytest.approx([0.38, 0.39])
	assert holding.local_individual == pytest.approx([0.21, 0.21])
	assert holding.scripless_ratio == pytest.approx([0.9712, 0.9720])


def test_broker_clustering_annotation_is_carried_through():
	payload = _payload(selected_brokers=["yp", "pd", "cc"], optimum_n_selected_cluster=2, optimum_corr=0.8123)
	clustering = payload.annotations.clustering
	assert clustering is not None
	assert clustering.selected_brokers == ["yp", "pd", "cc"]
	assert clustering.n_selected_cluster == 2
	assert clustering.correlation == pytest.approx(0.8123)


def test_foreign_analysis_has_no_clustering_and_switches_the_label():
	payload = _payload(analysis_method="foreign")
	assert payload.annotations.clustering is None
	assert payload.annotations.abbrev == "F"
	assert payload.annotations.method_label == "Foreign Flow"
	assert payload.summary.method == "foreign"


# ==========
# Missing values and JSON safety
# ==========
def test_missing_values_become_null_never_nan():
	frame = _indicators()
	frame.loc[frame.index[0], ["vwap", "prop", "pricecorrel", "close"]] = np.nan
	frame.loc[frame.index[1], "mf"] = np.inf
	payload = _payload(wf_indicators=frame)
	assert payload.indicators.vwap[0] is None
	assert payload.indicators.proportion[0] is None
	assert payload.price.close[0] is None
	assert payload.indicators.net_value[1] is None


def test_a_fully_missing_latest_row_yields_a_null_summary_not_an_error():
	frame = _indicators()
	frame.loc[frame.index[-1], ["close", "vwap", "mf", "prop", "netprop", "pricecorrel", "mapricecorrel"]] = np.nan
	frame["pow"] = np.nan
	summary = _payload(wf_indicators=frame).summary
	assert summary.close is None
	assert summary.vwap is None
	assert summary.money_flow is None
	assert summary.power is None
	assert summary.power_label is None
	assert summary.date == datetime.date(2026, 9, 18)


def test_absent_holding_composition_and_profile_are_empty_not_fabricated():
	payload = _payload(holding_composition=None, hist_bar=None, peaks_index=None)
	assert payload.holding_composition is None
	assert payload.profile_zones == []


def test_payload_serializes_to_json_with_native_scalars_and_iso_dates():
	raw = _payload(selected_brokers=["yp"], optimum_n_selected_cluster=2, optimum_corr=0.81).model_dump(mode="json")
	encoded = json.dumps(raw)  # raises on numpy scalars, NaN passes so check separately
	assert "NaN" not in encoded and "Infinity" not in encoded
	assert raw["price"]["dates"][0] == "2026-09-15"
	assert raw["summary"]["date"] == "2026-09-18"
	assert raw["period"]["start"] == "2026-09-15"
	assert raw["meta"]["generated_at"].startswith("2026-09-18T17:30:00")
	assert isinstance(raw["price"]["close"][0], float)
	assert isinstance(raw["summary"]["power"], int)


def test_empty_indicator_frame_is_rejected_rather_than_shipped_half_built():
	with pytest.raises(ValueError):
		_payload(wf_indicators=pd.DataFrame())
