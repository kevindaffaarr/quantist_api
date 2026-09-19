"""
Shared contract and criteria for the Whale (broker) and Foreign screeners.

Foreign and Whale keep separate data classes: their sources, indicators and
clustering differ and must not be merged. What they genuinely share is the
lifecycle (resolve defaults -> filter stockcodes -> load data -> rank) and the
criteria maths below, which is pure and driven by column names so neither
family has to know anything about the other.

DB dataparam
	- param: screener_status: 0 (not running), 1 (running)
	- param: screener_last_update: datetime, nan
	- param: screener_period: integer, 5
	- param: screener_minvalue: float, 0
	- param: screener_minfreq: integer, 0
	- param: screener_minwbuy: float, 0
	- param: screener_minwsell: float, 0
DB screener_result
	- date
	- screener_method
	- code
	- close
	- money flow
	- Proportion
	- correlation
	- vwap
"""
from __future__ import annotations

import abc
import asyncio
import datetime
import math
from typing import Any

import numpy as np
import pandas as pd

from .helper import Bin


class WhaleScreener(abc.ABC):
	"""
	What routers/whaleanalysis.py reads back off any screener, foreign or whale.

	screen() fills these in. startdate stays loosely typed on purpose: until the
	data is loaded it may still be an unresolved SQL subquery, which is why the
	router guards it with isinstance() before formatting it.
	"""

	startdate: Any
	enddate: datetime.date
	top_stockcodes: pd.DataFrame

	@abc.abstractmethod
	async def screen(self) -> WhaleScreener:
		"""Load, filter and rank; returns self."""


# ==========
# VWAP criteria: (code, date) indexed frames, close/vwap column names per family
# ==========
def vwap_rally(data: pd.DataFrame, close: str = "close", vwap: str = "vwap") -> list:
	"""Rally: close never dips below vwap within the window."""
	hit = (data[close] >= data[vwap]).groupby(level="code").all()
	return hit[hit].index.tolist()


def vwap_around(data: pd.DataFrame, percentage_range: float, close: str = "close", vwap: str = "vwap") -> list:
	"""Around VWAP: last close within +/- percentage_range of last vwap."""
	last = data[[close, vwap]].groupby(level="code").last()
	inside = (last[close] >= last[vwap] * (1 - percentage_range)) & (last[close] <= last[vwap] * (1 + percentage_range))
	return last[inside].index.tolist()


def _vwap_cross(data: pd.DataFrame, close: str, vwap: str, above: bool) -> list:
	"""
	Codes that are on the wanted side of vwap now and crossed onto it in-window.

	Replaces a rolling(2).apply() of a Python lambda: the cross is just "flagged
	now, not flagged on the previous bar". The first bar of each code has no
	previous bar, which the rolling version scored NaN and any() skipped, so it
	is seeded as already-flagged to stay out of the result.
	"""
	last = data[[close, vwap]].groupby(level="code").last()
	side = last[close] >= last[vwap] if above else last[close] <= last[vwap]
	candidates = last[side].index

	flag = data[close] >= data[vwap] if above else data[close] <= data[vwap]
	previous = flag.groupby(level="code").shift(1, fill_value=True)
	crossed = (flag & ~previous).groupby(level="code").any()
	return [code for code in candidates if crossed.get(code, False)]


def vwap_breakout(data: pd.DataFrame, close: str = "close", vwap: str = "vwap") -> list:
	"""Breakout: close crossed up through vwap in-window and is above it now."""
	return _vwap_cross(data, close, vwap, above=True)


def vwap_breakdown(data: pd.DataFrame, close: str = "close", vwap: str = "vwap") -> list:
	"""Breakdown: close crossed down through vwap in-window and is below it now."""
	return _vwap_cross(data, close, vwap, above=False)


# ==========
# Volume profile criteria
# ==========
VPROFILE_ROLES = ("support", "resistance", "undetermined")
VPROFILE_BEHAVIORS = ("acceptance", "rejection", "test", "breakout_up", "breakdown", "undetermined")


def vprofile_reading(closes: pd.Series, zones: pd.IntervalIndex, checking_period: int) -> dict[str, Any]:
	"""
	Read the net-value peak zone the last checking_period closes are working on.

	Pure and deterministic: zones in, closes in, native scalars out. Both
	families feed it their own profile, and every label below is read off the
	observed closes alone - nothing is inferred from the size or sign of the
	flow, and anything not positively observed stays "undetermined".

	Zone: the one holding the most recent close of the window; if the window
	touched none, the zone whose mid is nearest the last close, reported with
	in_zone False so the distance can still be read off it. Bins are (low, high]
	throughout, matching the pd.cut intervals the profile is built from.

	Role, from the approach direction only. The approach price is the most
	recent close at or before that touch which sits outside the zone; above the
	zone means price came down onto it (support), below means price came up
	into it (resistance). A zone price has never been observed outside of has
	no approach, so it keeps no role.

	Behavior, first rule that matches:
	  - no close inside the window              -> undetermined (nothing to read)
	  - last close inside, previous one too     -> acceptance (>=2 consecutive)
	  - last close inside, previous one outside -> test (one isolated touch)
	  - last close outside, window touched:
	      left above, approached from below     -> breakout_up
	      left above, approached from above     -> rejection (bounced off support)
	      left below, approached from above     -> breakdown
	      left below, approached from below     -> rejection (turned back at resistance)
	      no role                               -> undetermined
	"Previous close" is the bar before the last one whether or not the window
	reaches it, so a one-bar checking_period still sees a streak. touch_count
	counts the window's closes inside the zone; the distance is signed, positive
	when the last close sits above the zone mid.

	A missing (NaN) last close says nothing about now: role, behavior and the
	distance stay empty, while membership still answers for the window.
	"""
	reading: dict[str, Any] = {
		"vprofile_in_zone": False,
		"vprofile_zone_role": "undetermined",
		"vprofile_zone_behavior": "undetermined",
		"vprofile_zone_low": None,
		"vprofile_zone_high": None,
		"vprofile_zone_mid": None,
		"vprofile_distance_to_mid_pct": None,
		"vprofile_touch_count": 0,
	}
	values = closes.to_numpy(dtype="float64")
	if len(zones) == 0 or len(values) == 0:
		return reading

	# get_indexer scores a NaN close as -1 ("in no zone"), same as comparing it.
	window = values[-checking_period:]
	membership = zones.get_indexer(window)
	touched = np.flatnonzero(membership >= 0)
	reading["vprofile_in_zone"] = bool(len(touched))

	anchor = len(values) - len(window) + (touched[-1] if len(touched) else len(window) - 1)
	last = float(values[-1])
	if len(touched):
		selected = int(membership[touched[-1]])
	elif math.isnan(last):
		return reading  # never touched, and no price to measure the nearest zone from
	else:
		selected = int(np.abs(zones.mid.to_numpy() - last).argmin())

	low = float(zones.left.to_numpy()[selected])
	high = float(zones.right.to_numpy()[selected])
	mid = (low + high) / 2
	inside = (values > low) & (values <= high)
	reading.update({
		"vprofile_zone_low": low,
		"vprofile_zone_high": high,
		"vprofile_zone_mid": mid,
		"vprofile_touch_count": int(inside[-checking_period:].sum()),
	})
	if math.isnan(last):
		return reading
	reading["vprofile_distance_to_mid_pct"] = round((last - mid) / mid * 100, 4) if mid else None

	# Where price came from: the last close outside the zone, at or before the touch.
	approached = np.flatnonzero(~inside[:anchor + 1] & ~np.isnan(values[:anchor + 1]))
	if len(approached):
		reading["vprofile_zone_role"] = "support" if values[approached[-1]] > high else "resistance"

	role = reading["vprofile_zone_role"]
	if not reading["vprofile_touch_count"]:
		behavior = "undetermined"
	elif inside[-1]:
		behavior = "acceptance" if len(inside) > 1 and inside[-2] else "test"
	elif role == "undetermined":
		behavior = "undetermined"
	elif last > high:
		behavior = "breakout_up" if role == "resistance" else "rejection"
	else:
		behavior = "breakdown" if role == "support" else "rejection"
	reading["vprofile_zone_behavior"] = behavior

	return reading


async def vprofile_annotate(data: pd.DataFrame, checking_period: int) -> tuple[str, dict[str, Any]]:
	"""One code: fit its volume profile, then read the zone its recent closes work on."""
	code: str = data.index.get_level_values("code")[0]  # type: ignore

	bin_obj: Bin = await Bin(data=data).fit()
	zones = pd.IntervalIndex.from_tuples([]) \
		if bin_obj.nbins <= 2 or len(bin_obj.peaks_index) == 0 \
		else pd.IntervalIndex(bin_obj.hist_bar.index[bin_obj.peaks_index])

	return code, vprofile_reading(data["close"].astype(float), zones, checking_period)  # type: ignore


async def vprofile_annotations(data: pd.DataFrame, checking_period: int) -> pd.DataFrame:
	"""Per-code zone annotations, indexed by code, ready to join onto top_stockcodes."""
	results = await asyncio.gather(*[
		vprofile_annotate(group, checking_period)
		for _, group in data.groupby(level="code", group_keys=False)
	])
	return pd.DataFrame.from_dict(dict(results), orient="index").rename_axis("code")


async def vprofile_inside(data: pd.DataFrame, checking_period: int) -> tuple[str, bool]:
	"""
	Is any of the last checking_period closes sitting in a net-value peak zone?

	Membership only, unchanged: the annotations that come with it are read off
	the same profile but this answer stays the one the screeners rank on.
	"""
	code, reading = await vprofile_annotate(data, checking_period)
	return code, reading["vprofile_in_zone"]


async def vprofile_stocklist(data: pd.DataFrame, checking_period: int) -> list[str]:
	"""Codes whose recent closes sit inside their own net-value trading zone."""
	results = await asyncio.gather(*[
		vprofile_inside(group, checking_period)
		for _, group in data.groupby(level="code", group_keys=False)
	])
	inside = pd.Series(dict(results))
	return inside[inside].index.tolist()


def vprofile_behavior_codes(annotations: pd.DataFrame, behavior: str) -> list[str]:
	"""Codes whose current reading observed exactly this behavior. Pure, frame in, codes out."""
	return annotations.index[annotations["vprofile_zone_behavior"] == behavior].tolist()


def vprofile_role_behavior_codes(annotations: pd.DataFrame, role: str, behavior: str) -> list[str]:
	"""Codes whose current reading observed exactly this role and behavior."""
	selected = (
		(annotations["vprofile_zone_role"] == role)
		& (annotations["vprofile_zone_behavior"] == behavior)
	)
	return annotations.index[selected].tolist()


async def vprofile_breakout(data: pd.DataFrame, checking_period: int) -> list[str]:
	"""
	Codes that broke out above the zone they were working on.

	Same per-code profile vprofile_inside/vprofile_annotations read, filtered to
	the "breakout_up" behavior: within the last checking_period the closes
	touched the selected net-value peak zone, price had approached it from
	below (so the zone was acting as resistance), and the last close now sits
	above it. Point-in-time - only the window's own closes decide.

	Membership is not the signal: a code still inside its zone (acceptance,
	test) or one that touched and turned back (rejection) is not selected, and
	no signal is read off the zone's role alone.
	"""
	return vprofile_behavior_codes(await vprofile_annotations(data, checking_period), "breakout_up")


async def vprofile_breakdown(data: pd.DataFrame, checking_period: int) -> list[str]:
	"""
	Codes that broke down out of the zone they were working on.

	The mirror of vprofile_breakout, filtered to the "breakdown" behavior: the
	window touched the selected zone, price had approached it from above (so the
	zone was acting as support), and the last close now sits below it. Same
	exclusions - membership, an isolated test, a rejection, or a bare role is
	never a signal.
	"""
	return vprofile_behavior_codes(await vprofile_annotations(data, checking_period), "breakdown")


async def vprofile_support_bounce(data: pd.DataFrame, checking_period: int) -> list[str]:
	"""Codes that touched support and closed back above it, confirming a bounce."""
	return vprofile_role_behavior_codes(
		await vprofile_annotations(data, checking_period), "support", "rejection"
	)


async def vprofile_resistance_rejection(data: pd.DataFrame, checking_period: int) -> list[str]:
	"""Codes that touched resistance and closed back below it, confirming rejection."""
	return vprofile_role_behavior_codes(
		await vprofile_annotations(data, checking_period), "resistance", "rejection"
	)


# Keyed by dp.ScreenerList value; screener.py stays free of the dependencies import.
VPROFILE_CRITERIA = {
	"vprofile_inside": vprofile_stocklist,
	"vprofile_breakout": vprofile_breakout,
	"vprofile_breakdown": vprofile_breakdown,
	"vprofile_support_bounce": vprofile_support_bounce,
	"vprofile_resistance_rejection": vprofile_resistance_rejection,
}


async def vprofile_criteria_stocklist(
	data: pd.DataFrame,
	checking_period: int,
	criteria: str = "vprofile_inside",
	) -> list[str]:
	"""Select codes with the requested volume profile criterion, membership by default."""
	if criteria not in VPROFILE_CRITERIA:
		raise ValueError(f"Invalid screener_vprofile_criteria: {criteria}")
	return await VPROFILE_CRITERIA[criteria](data, checking_period)
