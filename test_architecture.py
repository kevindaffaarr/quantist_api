"""
Architecture guards for the Radar/Screener class bases.

Radar and Screener share a per-family base so the lifecycle (resolve defaults ->
filter stockcodes -> load data) is guaranteed common, while Foreign and Whale
stay separate families because their sources and algorithms differ. These tests
pin that shape so it cannot drift back into implicit inheritance.
"""
import abc
import inspect

import pytest

from quantist_library import brokerflow as bf
from quantist_library import foreignflow as ff
from quantist_library import screener as sc
from quantist_library import whaleflow as wf

FOREIGN_SCREENERS = [ff.ScreenerMoneyFlow, ff.ScreenerVWAP, ff.ScreenerVProfile]
WHALE_SCREENERS = [bf.ScreenerMoneyFlow, bf.ScreenerVWAP, bf.ScreenerVProfile]


@pytest.mark.parametrize("cls", FOREIGN_SCREENERS + WHALE_SCREENERS)
def test_every_screener_satisfies_the_common_contract(cls):
	assert issubclass(cls, sc.WhaleScreener)
	assert inspect.iscoroutinefunction(cls.screen)


@pytest.mark.parametrize(
	"radar, screener_base, family_base",
	[
		(ff.ForeignRadar, ff.ScreenerBase, ff.ForeignFlowBase),
		(bf.WhaleRadar, bf.ScreenerBase, bf.WhaleFlowBase),
	],
)
def test_radar_and_screener_share_their_family_base(radar, screener_base, family_base):
	assert issubclass(radar, family_base)
	assert issubclass(screener_base, family_base)
	# The shared lifecycle steps live on the base, not copied into each subclass.
	for step in ("_get_default_radar", "_get_stockcodes"):
		assert step in vars(family_base), f"{step} should be defined once on {family_base.__name__}"
		assert step not in vars(radar)
		assert step not in vars(screener_base)


@pytest.mark.parametrize("radar, screener_base", [(ff.ForeignRadar, ff.ScreenerBase), (bf.WhaleRadar, bf.ScreenerBase)])
def test_a_screener_is_not_a_radar(radar, screener_base):
	# The old code made every screener inherit the radar, dragging chart()/
	# y_axis_type along for the ride. Siblings, not parent/child.
	assert not issubclass(screener_base, radar)


@pytest.mark.parametrize("foreign, whale", [
	(ff.ForeignRadar, bf.WhaleRadar),
	(ff.ScreenerBase, bf.ScreenerBase),
	(ff.StockFFFull, bf.StockBFFull),
])
def test_foreign_and_whale_data_classes_stay_separate(foreign, whale):
	assert not issubclass(foreign, whale)
	assert not issubclass(whale, foreign)
	# Their only common ancestors may be the abstract contract and object.
	shared = (set(foreign.__mro__) & set(whale.__mro__)) - {object, abc.ABC, sc.WhaleScreener}
	assert shared == set(), f"unexpected shared base(s): {shared}"


@pytest.mark.parametrize("cls", [wf.ForeignFlow, wf.BrokerFlow])
def test_whaleflow_full_charts_expose_one_lifecycle(cls):
	assert inspect.iscoroutinefunction(cls.fit)
	assert inspect.iscoroutinefunction(cls.chart)
