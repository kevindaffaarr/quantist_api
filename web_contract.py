"""
Semantic chart contract for the Quantist web frontend.

The Telegram/public API answers with a rendered Plotly figure. The web frontend
draws its own chart, so it needs the *meaning* of each layer instead: price
bars, named indicator series, profile zones with explicit boundaries, holding
composition, and the annotations the header prints. Nothing here knows about
ECharts, Plotly, or any option syntax.

Everything below is pure: pandas objects in, JSON-safe Pydantic models out. No
database, no globals, no services, so the contract is testable on its own and
the route stays a thin caller. Values stay numeric and unformatted — ratios are
ratios (0.3342, not "33.42%"), money is money (-1e12, not "-1.00T") — because
formatting is a UI decision and rounding it here would be lossy.
"""
from __future__ import annotations

import datetime
import math
from collections.abc import Iterable
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel

SCHEMA_VERSION = "1.0"

AnalysisMethod = Literal["foreign", "broker"]
PowerLabel = Literal["low", "medium", "high"]

# ==========
# Instrument aliases
# ==========
# The backend stores the index as "composite"; Indonesian users know it as IHSG.
# One canonical code, many spellings, resolved in one place so the route, the
# cache key and the frontend all agree on it.
_ALIASES: dict[str, str] = {
	"composite": "composite",
	"ihsg": "composite",
	"ihsg/composite": "composite",
}
_INDEX_DISPLAY: dict[str, tuple[str, list[str]]] = {
	"composite": ("IHSG / COMPOSITE", ["IHSG", "COMPOSITE"]),
}
DEFAULT_INSTRUMENT = "composite"

_METHOD_LABEL: dict[str, tuple[str, str]] = {
	"foreign": ("F", "Foreign Flow"),
	"broker": ("W", "Whale Flow"),
}
_POWER_LABEL: dict[int, PowerLabel] = {3: "high", 2: "medium", 1: "low"}


# ==========
# JSON-safe scalar helpers
# ==========
def _num(value: Any) -> float | None:
	"""Any numpy/pandas/Decimal scalar to a finite float, or None. NaN is not JSON."""
	if value is None or value is pd.NaT:
		return None
	try:
		number = float(value)
	except (TypeError, ValueError):
		return None
	return number if math.isfinite(number) else None


def _int(value: Any) -> int | None:
	number = _num(value)
	return None if number is None else int(number)


def _date(value: Any) -> datetime.date:
	"""Any date-like index value to a plain date. A missing date breaks the axis, so it raises."""
	if isinstance(value, datetime.datetime):  # covers pd.Timestamp
		return value.date()
	if isinstance(value, datetime.date):
		return value
	timestamp = pd.Timestamp(value)
	if timestamp is pd.NaT:
		raise ValueError(f"Date axis contains a missing value: {value!r}")
	return datetime.date(int(timestamp.year), int(timestamp.month), int(timestamp.day))


def _column(frame: pd.DataFrame, name: str) -> list[float | None]:
	"""One aligned series of JSON-safe numbers; a column the source never filled is all-null."""
	if name not in frame.columns:
		return [None] * len(frame)
	return [_num(value) for value in frame[name].to_numpy()]


def _last(frame: pd.DataFrame, name: str) -> float | None:
	return None if name not in frame.columns or frame.empty else _num(frame[name].iloc[-1])


# ==========
# Models
# ==========
class Instrument(BaseModel):
	code: str
	symbol: str
	display_name: str
	kind: Literal["index", "stock"]
	aliases: list[str] = []


class Period(BaseModel):
	start: datetime.date
	end: datetime.date
	bars: int
	params: dict[str, int | None] = {}


class Summary(BaseModel):
	"""The header line of the chart, numeric. 'W Proportion 33.42%' is proportion=0.3342."""
	date: datetime.date
	close: float | None = None
	vwap: float | None = None
	money_flow: float | None = None
	value_flow: float | None = None
	proportion: float | None = None
	net_proportion: float | None = None
	price_correlation: float | None = None
	ma_price_correlation: float | None = None
	power: int | None = None
	power_label: PowerLabel | None = None
	method: AnalysisMethod


class Price(BaseModel):
	"""Column-oriented OHLC; every list is aligned index-for-index with ``dates``."""
	dates: list[datetime.date]
	open: list[float | None]
	high: list[float | None]
	low: list[float | None]
	close: list[float | None]
	volume: list[float | None]
	value: list[float | None]


class Indicators(BaseModel):
	"""Named layers, aligned to ``price.dates``. Ratios are ratios, money is money."""
	vwap: list[float | None]
	value_flow: list[float | None]
	proportion: list[float | None]
	net_proportion: list[float | None]
	net_value: list[float | None]


class ProfileZone(BaseModel):
	"""One net-value profile bin. Boundaries are explicit: mid is derived, never guessed back."""
	low: float
	high: float
	mid: float
	net_value: float | None = None
	is_peak: bool = False


class HoldingComposition(BaseModel):
	periods: list[datetime.date]
	foreign: list[float | None]
	local_institutional: list[float | None]
	local_individual: list[float | None]
	scripless_ratio: list[float | None]


class Clustering(BaseModel):
	n_selected_cluster: int | None = None
	correlation: float | None = None
	selected_brokers: list[str] = []


class Annotations(BaseModel):
	method: AnalysisMethod
	method_label: str
	abbrev: str
	power_label: PowerLabel | None = None
	clustering: Clustering | None = None


class Meta(BaseModel):
	schema_version: str = SCHEMA_VERSION
	generated_at: datetime.datetime
	source: str = "quantist_api"


# ==========
# Screener metadata and results
# ==========
class ScreenerDefinition(BaseModel):
	"""
	What a screener is, and where its results come from.

	`results_endpoint` is the browser-facing web route; `legacy_endpoint` is
	the DataFrame-dump route Telegram still uses. `web_results_available` says
	whether the typed route can actually answer, so a client never implies it
	has results it cannot fetch.
	"""
	slug: str
	label: str
	group: str
	methods: list[AnalysisMethod]
	results_endpoint: str
	legacy_endpoint: str = ""
	web_results_available: bool = True


class ScreenerCatalog(BaseModel):
	schema_version: str = SCHEMA_VERSION
	screeners: list[ScreenerDefinition] = []


class ScreenerRow(BaseModel):
	"""
	One ranked stock.

	The columns the legacy screeners emit differ per criterion, so the ones
	every criterion shares are named fields and everything else rides in
	`extras`. Dropping the rest would lose the volume-profile annotations that
	are the whole point of those criteria.

	`rank` records the screener's own ordering. The screener already sorted
	these — by flow, by cross freshness, by node prominence, depending on the
	criterion — and that ordering is the answer. Recording it means a client
	can sort by any column and still get back to what the screener said.
	"""
	rank: int
	code: str
	display_name: str
	close: float | None = None
	money_flow: float | None = None
	proportion: float | None = None
	price_correlation: float | None = None
	vwap: float | None = None
	extras: dict[str, float | str | bool | None] = {}


class ScreenerPeriod(BaseModel):
	start: datetime.date | None = None
	end: datetime.date | None = None
	bars: int | None = None


class ScreenerResults(BaseModel):
	schema_version: str = SCHEMA_VERSION
	slug: str
	method: AnalysisMethod
	period: ScreenerPeriod
	count: int
	rows: list[ScreenerRow] = []
	meta: Meta


class WebChart(BaseModel):
	schema_version: str = SCHEMA_VERSION
	instrument: Instrument
	period: Period
	summary: Summary
	price: Price
	indicators: Indicators
	profile_zones: list[ProfileZone] = []
	holding_composition: HoldingComposition | None = None
	annotations: Annotations
	meta: Meta


# ==========
# Builders
# ==========
_SCREENER_GROUPS: dict[str, tuple[str, str, str]] = {
	# slug: (label, group, legacy results path under /whaleanalysis/screener/{method})
	"most_accumulated": ("Most accumulated", "Money flow", "top-money-flow"),
	"most_distributed": ("Most distributed", "Money flow", "top-money-flow"),
	"vwap_rally": ("Rally", "VWAP", "vwap"),
	"vwap_around": ("Around VWAP", "VWAP", "vwap"),
	"vwap_breakout": ("Breakout", "VWAP", "vwap"),
	"vwap_breakdown": ("Breakdown", "VWAP", "vwap"),
	"vprofile_inside": ("Inside zone", "Volume profile", "vprofile"),
	"vprofile_breakout": ("Zone breakout", "Volume profile", "vprofile"),
	"vprofile_breakdown": ("Zone breakdown", "Volume profile", "vprofile"),
	"vprofile_support_bounce": ("Support bounce", "Volume profile", "vprofile"),
	"vprofile_resistance_rejection": ("Resistance rejection", "Volume profile", "vprofile"),
}


# Which screener class answers a slug. The web route dispatches on this alone,
# so a backend slug missing here would be a screener the web surface cannot ask
# for — test_web_screener.py pins the two sets equal.
SCREENER_FAMILY: dict[str, str] = {
	"most_accumulated": "money_flow",
	"most_distributed": "money_flow",
	"vwap_rally": "vwap",
	"vwap_around": "vwap",
	"vwap_breakout": "vwap",
	"vwap_breakdown": "vwap",
	"vprofile_inside": "vprofile",
	"vprofile_breakout": "vprofile",
	"vprofile_breakdown": "vprofile",
	"vprofile_support_bounce": "vprofile",
	"vprofile_resistance_rejection": "vprofile",
}

# DataFrame column -> contract field. Anything else becomes an extra.
_SCREENER_FIELDS: dict[str, str] = {
	"close": "close",
	"mf": "money_flow",
	"prop": "proportion",
	"pricecorrel": "price_correlation",
	"corr": "price_correlation",
	"vwap": "vwap",
}


def _scalar(value: Any) -> float | str | bool | None:
	"""An extras value: a finite number, a plain string, a bool, or null."""
	if isinstance(value, bool):
		return value
	if isinstance(value, str):
		return value
	number = _num(value)
	if number is not None:
		return number
	return None


def _screener_date(value: Any) -> datetime.date | None:
	if value in (None, ""):
		return None
	try:
		return _date(value)
	except (ValueError, TypeError):
		return None


def build_screener_results(
	slug: str,
	method: AnalysisMethod,
	frame: pd.DataFrame,
	metadata: dict[str, Any] | None = None,
	generated_at: datetime.datetime | None = None,
	) -> ScreenerResults:
	"""Rank frame plus screener metadata to the browser-facing contract."""
	if slug not in SCREENER_FAMILY:
		raise ValueError(f"Unknown screener slug: {slug!r}")

	info = metadata or {}
	rows: list[ScreenerRow] = []
	if frame is not None and not frame.empty:
		# enumerate over the frame as given: the row order IS the ranking, so
		# nothing here re-sorts and nothing renumbers.
		for position, (code, record) in enumerate(frame.to_dict(orient="index").items(), start=1):
			instrument = resolve_instrument(str(code))
			named: dict[str, float | None] = {}
			extras: dict[str, float | str | bool | None] = {}
			for column, value in record.items():
				field = _SCREENER_FIELDS.get(str(column))
				if field is not None:
					named.setdefault(field, _num(value))
				else:
					extras[str(column)] = _scalar(value)
			rows.append(ScreenerRow(
				rank=position,
				code=instrument.code,
				display_name=instrument.display_name,
				extras=extras,
				**named,
			))

	return ScreenerResults(
		slug=slug,
		method=method,
		period=ScreenerPeriod(
			start=_screener_date(info.get("startdate")),
			end=_screener_date(info.get("enddate")),
			bars=_int(info.get("bar_range")),
		),
		count=len(rows),
		rows=rows,
		meta=Meta(generated_at=generated_at or datetime.datetime.now(datetime.UTC)),
	)


def build_screener_catalog(slugs: Iterable[str]) -> ScreenerCatalog:
	"""
	Screener metadata from the backend's own ScreenerList values.

	Each entry names the browser-facing results route, the legacy route
	Telegram still uses, and whether the typed results can be fetched. The
	criteria come from ScreenerList, so this cannot drift from what the
	backend supports.
	"""
	definitions: list[ScreenerDefinition] = []
	for slug in slugs:
		label, group, path = _SCREENER_GROUPS.get(slug, (slug.replace("_", " ").capitalize(), "Other", ""))
		definitions.append(ScreenerDefinition(
			slug=slug,
			label=label,
			group=group,
			methods=["foreign", "broker"],
			results_endpoint=f"/web-api/v1/screener/{slug}?method={{method}}",
			legacy_endpoint=f"/whaleanalysis/screener/{{method}}/{path}" if path else "",
			# True because /web-api/v1/screener/{slug} exists and answers; a slug
			# the dispatch table does not know could not be routed at all.
			web_results_available=slug in SCREENER_FAMILY,
		))
	return ScreenerCatalog(screeners=definitions)


def resolve_instrument(code: str | None) -> Instrument:
	"""Canonicalize a requested code. Empty/unknown-case input falls back to COMPOSITE."""
	requested = (code or "").strip().lower()
	canonical = _ALIASES.get(requested, requested or DEFAULT_INSTRUMENT)
	if canonical in _INDEX_DISPLAY:
		display_name, aliases = _INDEX_DISPLAY[canonical]
		return Instrument(code=canonical, symbol=canonical.upper(), display_name=display_name, kind="index", aliases=aliases)
	return Instrument(code=canonical, symbol=canonical.upper(), display_name=canonical.upper(), kind="stock")


def _profile_zones(hist_bar: pd.Series | None, peaks_index: Iterable[int] | None) -> list[ProfileZone]:
	if hist_bar is None or len(hist_bar) == 0:
		return []
	intervals = pd.IntervalIndex(hist_bar.index)  # type: ignore[arg-type]
	peaks = {int(index) for index in (peaks_index or [])}
	zones: list[ProfileZone] = []
	for position, (interval, value) in enumerate(zip(intervals, hist_bar.to_numpy())):
		low, high = float(interval.left), float(interval.right)
		if not (math.isfinite(low) and math.isfinite(high)):
			continue
		zones.append(ProfileZone(low=low, high=high, mid=(low + high) / 2, net_value=_num(value), is_peak=position in peaks))
	return zones


def _holding(holding_composition: pd.DataFrame | None) -> HoldingComposition | None:
	if holding_composition is None or holding_composition.empty:
		return None
	return HoldingComposition(
		periods=[_date(value) for value in holding_composition.index],
		foreign=_column(holding_composition, "foreign"),
		local_institutional=_column(holding_composition, "local_institutional"),
		local_individual=_column(holding_composition, "local_individual"),
		scripless_ratio=_column(holding_composition, "scripless_ratio"),
	)


def build_web_chart(
	code: str | None,
	wf_indicators: pd.DataFrame,
	analysis_method: AnalysisMethod,
	periods: dict[str, int | None] | None = None,
	startdate: datetime.date | None = None,
	enddate: datetime.date | None = None,
	hist_bar: pd.Series | None = None,
	peaks_index: Iterable[int] | None = None,
	holding_composition: pd.DataFrame | None = None,
	selected_brokers: list[str] | None = None,
	optimum_n_selected_cluster: int | None = None,
	optimum_corr: float | None = None,
	generated_at: datetime.datetime | None = None,
	) -> WebChart:
	"""Normalize a fitted whale/foreign flow into the versioned web chart contract."""
	if wf_indicators is None or wf_indicators.empty:
		raise ValueError("wf_indicators is empty: there is no chart to serialize.")

	frame = wf_indicators
	dates = [_date(value) for value in frame.index]
	power = _int(_last(frame, "pow"))
	abbrev, method_label = _METHOD_LABEL[analysis_method]

	return WebChart(
		instrument=resolve_instrument(code),
		period=Period(
			start=startdate or dates[0],
			end=enddate or dates[-1],
			bars=len(dates),
			params=dict(periods or {}),
		),
		summary=Summary(
			date=dates[-1],
			close=_last(frame, "close"),
			vwap=_last(frame, "vwap"),
			money_flow=_last(frame, "mf"),
			value_flow=_last(frame, "valflow"),
			proportion=_last(frame, "prop"),
			net_proportion=_last(frame, "netprop"),
			price_correlation=_last(frame, "pricecorrel"),
			ma_price_correlation=_last(frame, "mapricecorrel"),
			power=power,
			power_label=_POWER_LABEL.get(power) if power is not None else None,
			method=analysis_method,
		),
		price=Price(
			dates=dates,
			open=_column(frame, "openprice"),
			high=_column(frame, "high"),
			low=_column(frame, "low"),
			close=_column(frame, "close"),
			volume=_column(frame, "volume"),
			value=_column(frame, "value"),
		),
		indicators=Indicators(
			vwap=_column(frame, "vwap"),
			value_flow=_column(frame, "valflow"),
			proportion=_column(frame, "prop"),
			net_proportion=_column(frame, "netprop"),
			net_value=_column(frame, "mf"),
		),
		profile_zones=_profile_zones(hist_bar, peaks_index),
		holding_composition=_holding(holding_composition),
		annotations=Annotations(
			method=analysis_method,
			method_label=method_label,
			abbrev=abbrev,
			power_label=_POWER_LABEL.get(power) if power is not None else None,
			clustering=None if selected_brokers is None else Clustering(
				n_selected_cluster=_int(optimum_n_selected_cluster),
				correlation=_num(optimum_corr),
				selected_brokers=[str(broker) for broker in selected_brokers],
			),
		),
		meta=Meta(generated_at=generated_at or datetime.datetime.now(datetime.UTC)),
	)


__all__ = [
	"DEFAULT_INSTRUMENT",
	"SCHEMA_VERSION",
	"Instrument",
	"ScreenerResults",
	"ScreenerCatalog",
	"WebChart",
	"SCREENER_FAMILY",
	"build_screener_results",
	"build_screener_catalog",
	"build_web_chart",
	"resolve_instrument",
]
