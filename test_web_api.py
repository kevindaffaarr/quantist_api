"""
Wiring guards for the web chart route.

No server, no database: the app object is built at import time, so route
registration, the auth policy and the flow-to-contract read can all be checked
statically. What must not drift:

	* the new /web-api/v1 route exists and is versioned,
	* every pre-existing Telegram/public path is still there, untouched,
	* the web route sits behind the same API key as everything else, so the
	  browser can never be the direct caller.
"""
import datetime
from typing import Any

import numpy as np
import pandas as pd
import pytest

import main
import web_contract as wc
from routers import web_api

PATHS = main.app.openapi()["paths"]

EXISTING_PATHS = {
	"/param/dataparam",
	"/param/list/{list_category}",
	"/whaleanalysis",
	"/whaleanalysis/",
	"/whaleanalysis/chart",
	"/whaleanalysis/chart/foreign",
	"/whaleanalysis/chart/broker",
	"/whaleanalysis/radar",
	"/whaleanalysis/radar/foreign",
	"/whaleanalysis/radar/broker",
	"/whaleanalysis/full-data",
	"/whaleanalysis/full-data/foreign",
	"/whaleanalysis/full-data/broker",
	"/whaleanalysis/screener",
	"/whaleanalysis/screener/foreign",
	"/whaleanalysis/screener/foreign/top-money-flow",
	"/whaleanalysis/screener/foreign/vwap",
	"/whaleanalysis/screener/foreign/vprofile",
	"/whaleanalysis/screener/broker/top-money-flow",
	"/whaleanalysis/screener/broker/vwap",
	"/whaleanalysis/screener/broker/vprofile",
}


def test_the_web_route_is_registered_and_versioned():
	assert "/web-api/v1/chart/{code}" in PATHS


def test_the_code_less_route_defaults_to_composite():
	# The web root is a COMPOSITE page, so the API answers COMPOSITE when asked
	# for no instrument at all.
	assert "/web-api/v1/chart" in PATHS
	assert wc.resolve_instrument(None).code == wc.DEFAULT_INSTRUMENT == "composite"


def test_adding_the_web_route_left_every_existing_path_in_place():
	assert EXISTING_PATHS <= set(PATHS)


def test_the_web_route_keeps_the_same_api_key_policy():
	# get_api_key is applied by main.include_router, so it shows up as the
	# operation's security requirement, same as every other route.
	def scheme_names(path):
		return {name for op in PATHS[path].values() for requirement in op.get("security", []) for name in requirement}

	assert scheme_names("/web-api/v1/chart/{code}") == scheme_names("/whaleanalysis/chart")
	assert scheme_names("/web-api/v1/chart/{code}")


class _StubFlow:
	"""Stands in for a fitted BrokerFlow: only the attributes the reader touches."""

	class _Method:
		value = "broker"

	class _Bin:
		hist_bar = pd.Series(
			[-2.0e11, 5.0e11],
			index=pd.IntervalIndex.from_breaks([6300.0, 6450.0, 6600.0]),
		)
		peaks_index = [1]

	analysis_method: Any = _Method()
	stockcode = "composite"
	startdate = datetime.date(2026, 6, 18)
	enddate = datetime.date(2026, 9, 18)
	period_mf = 1
	period_prop = 10
	period_pricecorrel = 10
	period_mapricecorrel = 100
	period_vwap = 21
	bin_obj: Any = _Bin()
	selected_broker: Any = ["yp", "pd"]
	optimum_n_selected_cluster = 2
	optimum_corr = 0.7431
	holding_composition: Any = pd.DataFrame(
		{"foreign": [0.4], "local_institutional": [0.39], "local_individual": [0.21], "scripless_ratio": [0.97]},
		index=pd.to_datetime([datetime.date(2026, 8, 1)]),
	)
	wf_indicators = pd.DataFrame(
		{
			"openprice": [6400.0, 6430.0],
			"high": [6450.0, 6460.0],
			"low": [6380.0, 6400.0],
			"close": [6430.0, 6441.0],
			"value": [9.1e12, 9.4e12],
			"volume": [1.7e10, 1.8e10],
			"mf": [-9.0e11, -1.0e12],
			"valflow": [-2.1e12, -3.1e12],
			"prop": [0.31, 0.3342],
			"netprop": [0.11, 0.1221],
			"pricecorrel": [0.39, 0.4129],
			"mapricecorrel": [0.61, 0.6155],
			"vwap": [6560.0, 6569.35],
			"pow": [2, 2],
		},
		index=pd.to_datetime([datetime.date(2026, 9, 17), datetime.date(2026, 9, 18)]),
	)


def test_a_fitted_flow_reads_into_the_contract():
	payload = web_api.build_payload(_StubFlow(), requested_code="IHSG")

	assert payload.instrument.code == "composite"
	assert payload.instrument.display_name == "IHSG / COMPOSITE"
	assert payload.summary.close == pytest.approx(6441.0)
	assert payload.summary.vwap == pytest.approx(6569.35)
	assert payload.summary.money_flow == pytest.approx(-1.0e12)
	assert payload.period.params == {"mf": 1, "prop": 10, "pricecorrel": 10, "mapricecorrel": 100, "vwap": 21}
	assert [zone.is_peak for zone in payload.profile_zones] == [False, True]
	assert payload.annotations.clustering is not None
	assert payload.annotations.clustering.selected_brokers == ["yp", "pd"]
	assert payload.holding_composition is not None


def test_a_foreign_flow_without_clustering_or_profile_still_serializes():
	flow = _StubFlow()
	flow.analysis_method = type("M", (), {"value": "foreign"})()
	flow.selected_broker = None
	flow.bin_obj = None
	flow.holding_composition = None

	payload = web_api.build_payload(flow, requested_code=None)

	assert payload.annotations.clustering is None
	assert payload.profile_zones == []
	assert payload.holding_composition is None
	assert payload.instrument.code == "composite"


def test_the_payload_a_flow_produces_is_json_safe():
	flow = _StubFlow()
	flow.wf_indicators = flow.wf_indicators.copy()
	flow.wf_indicators.loc[flow.wf_indicators.index[0], "vwap"] = np.nan

	raw = web_api.build_payload(flow).model_dump(mode="json")

	assert raw["schema_version"] == wc.SCHEMA_VERSION
	assert raw["indicators"]["vwap"][0] is None
	assert raw["price"]["dates"] == ["2026-09-17", "2026-09-18"]
