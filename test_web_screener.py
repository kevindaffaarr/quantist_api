"""
Browser-facing screener results.

The legacy /whaleanalysis/screener routes answer with a DataFrame dump whose
columns differ per criterion. The web surface needs one stable shape it can
render without knowing which criterion produced it, so the normalizer maps the
columns it knows by name and carries the rest through as extras rather than
dropping them.

Pure: frames in, JSON-safe models out. No database, no services.
"""
import datetime
import json

import numpy as np
import pandas as pd
import pytest

import dependencies as dp
import web_contract as wc


def _frame() -> pd.DataFrame:
	return pd.DataFrame(
		{
			"close": [6441.0, 1250.0],
			"mf": [-1.0e12, 4.2e10],
			"prop": [0.3342, 0.1810],
			"pricecorrel": [0.4129, -0.2],
			"vwap": [6569.35, 1199.5],
			"vprofile_zone_low": [6300.0, None],
			"vprofile_zone_behavior": ["rejection", None],
		},
		index=pd.Index(["composite", "bbri"], name="code"),
	)


def _results(**overrides):
	kwargs: dict = {
		"slug": "vwap_rally",
		"method": "broker",
		"frame": _frame(),
		"metadata": {"bar_range": 5, "startdate": "2026-09-01", "enddate": "2026-09-18"},
		"generated_at": datetime.datetime(2026, 9, 18, 17, 30, tzinfo=datetime.UTC),
	}
	kwargs.update(overrides)
	return wc.build_screener_results(**kwargs)


def test_every_backend_slug_is_routable_and_knows_its_family():
	# The web route dispatches on slug alone, so a slug the backend accepts and
	# the web layer does not would be a silently missing screener.
	assert set(wc.SCREENER_FAMILY) == {entry.value for entry in dp.ScreenerList}
	assert set(wc.SCREENER_FAMILY.values()) == {"money_flow", "vwap", "vprofile"}
	assert wc.SCREENER_FAMILY["most_accumulated"] == "money_flow"
	assert wc.SCREENER_FAMILY["vwap_breakdown"] == "vwap"
	assert wc.SCREENER_FAMILY["vprofile_support_bounce"] == "vprofile"


def test_results_carry_the_slug_method_and_row_count():
	results = _results()
	assert results.schema_version == wc.SCHEMA_VERSION
	assert results.slug == "vwap_rally"
	assert results.method == "broker"
	assert results.count == 2
	assert [row.code for row in results.rows] == ["composite", "bbri"]


def test_known_columns_become_named_numeric_fields():
	row = _results().rows[0]
	assert row.close == pytest.approx(6441.0)
	assert row.money_flow == pytest.approx(-1.0e12)
	assert row.proportion == pytest.approx(0.3342)
	assert row.price_correlation == pytest.approx(0.4129)
	assert row.vwap == pytest.approx(6569.35)


def test_the_index_display_name_is_resolved_like_everywhere_else():
	rows = {row.code: row for row in _results().rows}
	assert rows["composite"].display_name == "IHSG / COMPOSITE"
	assert rows["bbri"].display_name == "BBRI"


def test_unknown_columns_survive_as_extras_rather_than_being_dropped():
	extras = _results().rows[0].extras
	assert extras["vprofile_zone_low"] == pytest.approx(6300.0)
	assert extras["vprofile_zone_behavior"] == "rejection"
	# Named fields are not duplicated into extras.
	assert "close" not in extras and "mf" not in extras


def test_missing_values_are_null_never_nan():
	frame = _frame()
	frame.loc["bbri", ["close", "vwap", "mf"]] = np.nan
	frame.loc["bbri", "pricecorrel"] = np.inf
	rows = {row.code: row for row in _results(frame=frame).rows}

	assert rows["bbri"].close is None
	assert rows["bbri"].vwap is None
	assert rows["bbri"].money_flow is None
	assert rows["bbri"].price_correlation is None
	assert rows["bbri"].extras["vprofile_zone_low"] is None


def test_an_empty_result_is_an_empty_list_not_an_error():
	results = _results(frame=pd.DataFrame())
	assert results.count == 0
	assert results.rows == []


def test_the_period_comes_from_the_screener_metadata():
	period = _results().period
	assert period.start == datetime.date(2026, 9, 1)
	assert period.end == datetime.date(2026, 9, 18)
	assert period.bars == 5


def test_a_null_startdate_in_the_metadata_is_tolerated():
	period = _results(metadata={"bar_range": 5, "startdate": None, "enddate": "2026-09-18"}).period
	assert period.start is None
	assert period.end == datetime.date(2026, 9, 18)


def test_results_serialize_to_json_with_native_scalars():
	raw = _results().model_dump(mode="json")
	encoded = json.dumps(raw)
	assert "NaN" not in encoded and "Infinity" not in encoded
	assert raw["period"]["end"] == "2026-09-18"
	assert raw["meta"]["generated_at"].startswith("2026-09-18T17:30:00")
	assert isinstance(raw["rows"][0]["close"], float)


def test_an_unknown_slug_is_rejected_rather_than_guessed():
	with pytest.raises(ValueError):
		_results(slug="not_a_screener")
