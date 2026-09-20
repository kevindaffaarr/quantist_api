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
# Instrument catalogue and screener metadata
# ==========
class InstrumentListItem(BaseModel):
	"""One row of the instrument dropdown, already resolved for display."""
	code: str
	symbol: str
	display_name: str
	kind: Literal["index", "stock"]


class InstrumentList(BaseModel):
	schema_version: str = SCHEMA_VERSION
	category: Literal["stock", "index", "broker"]
	count: int
	instruments: list[InstrumentListItem] = []


class ScreenerDefinition(BaseModel):
	"""
	What a screener is, not what it currently returns.

	`results_endpoint` names the legacy route that produces the list today;
	`web_results_available` is False until a typed web-api equivalent exists,
	so a client can show the criterion without implying it has the results.
	"""
	slug: str
	label: str
	group: str
	methods: list[AnalysisMethod]
	results_endpoint: str
	web_results_available: bool = False


class ScreenerCatalog(BaseModel):
	schema_version: str = SCHEMA_VERSION
	screeners: list[ScreenerDefinition] = []


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


def build_instrument_list(category: str, codes: Iterable[str]) -> InstrumentList:
	"""
	Database codes to the dropdown contract. Pure: rows in, models out.

	Every code goes through the same resolver the chart route uses, so the
	dropdown and the chart agree on COMPOSITE/IHSG without a second table.
	"""
	items: list[InstrumentListItem] = []
	seen: set[str] = set()
	for raw in codes:
		instrument = resolve_instrument(str(raw))
		if instrument.code in seen:
			continue
		seen.add(instrument.code)
		items.append(InstrumentListItem(
			code=instrument.code,
			symbol=instrument.symbol,
			display_name=instrument.display_name,
			kind=instrument.kind,
		))
	items.sort(key=lambda item: item.code)
	return InstrumentList(category=category, count=len(items), instruments=items)  # type: ignore[arg-type]


def build_screener_catalog(slugs: Iterable[str]) -> ScreenerCatalog:
	"""
	Screener metadata from the backend's own ScreenerList values.

	Deliberately metadata only: the criteria are real and the legacy endpoint
	that answers them is named, but `web_results_available` stays False until
	a typed web-api result contract exists. A client that renders this cannot
	accidentally present a criterion as a result.
	"""
	definitions: list[ScreenerDefinition] = []
	for slug in slugs:
		label, group, path = _SCREENER_GROUPS.get(slug, (slug.replace("_", " ").capitalize(), "Other", ""))
		definitions.append(ScreenerDefinition(
			slug=slug,
			label=label,
			group=group,
			methods=["foreign", "broker"],
			results_endpoint=f"/whaleanalysis/screener/{{method}}/{path}" if path else "",
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
	"InstrumentList",
	"ScreenerCatalog",
	"WebChart",
	"build_instrument_list",
	"build_screener_catalog",
	"build_web_chart",
	"resolve_instrument",
]
