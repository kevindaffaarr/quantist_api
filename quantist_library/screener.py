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
from typing import Any

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
async def vprofile_inside(data: pd.DataFrame, checking_period: int) -> tuple[str, bool]:
	"""Is any of the last checking_period closes sitting in a net-value peak zone?"""
	code: str = data.index.get_level_values("code")[0]  # type: ignore

	bin_obj: Bin = await Bin(data=data).fit()
	if bin_obj.nbins <= 2:
		return code, False

	trading_zone = bin_obj.hist_bar.index[bin_obj.peaks_index]
	last_close = data["close"].iloc[-checking_period:]
	is_inside_interval = any(last_close.apply(lambda x: any(x in interval for interval in trading_zone)))

	return code, is_inside_interval


async def vprofile_stocklist(data: pd.DataFrame, checking_period: int) -> list[str]:
	"""Codes whose recent closes sit inside their own net-value trading zone."""
	results = await asyncio.gather(*[
		vprofile_inside(group, checking_period)
		for _, group in data.groupby(level="code", group_keys=False)
	])
	inside = pd.Series(dict(results))
	return inside[inside].index.tolist()
