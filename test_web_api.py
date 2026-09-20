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
import json
from typing import Any

import numpy as np
import pandas as pd
import pytest

import dependencies as dp
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


# ==========
# Analysis method
# ==========
def test_the_chart_route_takes_a_method_and_defaults_to_broker():
	# Foreign and Whale are two different analyses. The web UI switches between
	# them with this parameter, so it has to be a real, documented query param.
	parameters = {entry["name"]: entry for entry in PATHS["/web-api/v1/chart/{code}"]["get"]["parameters"]}
	assert "method" in parameters
	assert parameters["method"]["in"] == "query"

	schema = parameters["method"]["schema"]
	allowed = schema.get("enum") or schema.get("allOf", [{}])[0].get("enum") or []
	if not allowed:
		allowed = [value.value for value in dp.AnalysisMethod]
	assert set(allowed) == {"foreign", "broker"}
	assert schema.get("default", dp.AnalysisMethod.broker.value) == "broker"


@pytest.mark.parametrize("method, expected_label, abbrev", [("foreign", "Foreign Flow", "F"), ("broker", "Whale Flow", "W")])
def test_each_method_labels_its_own_payload(method, expected_label, abbrev):
	flow = _StubFlow()
	flow.analysis_method = type("M", (), {"value": method})()
	if method == "foreign":
		flow.selected_broker = None

	payload = web_api.build_payload(flow)

	assert payload.summary.method == method
	assert payload.annotations.method_label == expected_label
	assert payload.annotations.abbrev == abbrev


def test_the_worker_forwards_exactly_the_parameters_this_route_accepts():
	# The worker allow-lists query parameters; anything it forwards has to
	# exist here or the origin would 422 on a valid-looking request.
	accepted = {entry["name"] for entry in PATHS["/web-api/v1/chart/{code}"]["get"]["parameters"]}
	assert {"method", "startdate", "enddate", "clustering_method"} <= accepted


# ==========
# Screener metadata
# ==========
def test_the_screener_routes_are_registered_behind_the_same_key():
	def scheme_names(path):
		return {name for op in PATHS[path].values() for requirement in op.get("security", []) for name in requirement}

	for path in ("/web-api/v1/screeners", "/web-api/v1/screener/{slug}"):
		assert path in PATHS
		assert scheme_names(path) == scheme_names("/whaleanalysis/chart")


def test_the_browser_reads_code_lists_from_the_existing_param_route():
	# The worker proxies /param/list/{category} for the dropdown; no second
	# catalogue endpoint is part of that path.
	assert "/param/list/{list_category}" in PATHS
	assert "/web-api/v1/instruments" not in PATHS


def test_screener_catalog_covers_every_backend_criterion():
	catalog = wc.build_screener_catalog(entry.value for entry in dp.ScreenerList)

	assert {entry.slug for entry in catalog.screeners} == {entry.value for entry in dp.ScreenerList}
	for entry in catalog.screeners:
		assert entry.label and entry.group
		assert entry.methods == ["foreign", "broker"]
		# Every criterion is routable on the browser-facing path now.
		assert entry.web_results_available is True
		assert entry.results_endpoint == f"/web-api/v1/screener/{entry.slug}?method={{method}}"
		# The legacy route is still named, and still unchanged for Telegram.
		assert entry.legacy_endpoint.startswith("/whaleanalysis/screener/")


def test_screener_catalog_is_json_safe():
	raw = wc.build_screener_catalog(["vwap_rally"]).model_dump(mode="json")
	assert json.dumps(raw)
	assert raw["screeners"][0]["label"] == "Rally"
	assert raw["screeners"][0]["group"] == "VWAP"
