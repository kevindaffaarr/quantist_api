"""
Versioned read-only chart endpoint for the Quantist web frontend.

Separate from /whaleanalysis on purpose: that router answers with a rendered
Plotly figure and Telegram depends on its exact response shape, so nothing here
touches it. This router answers with the semantic contract in web_contract.py
and is versioned in its path, so the frontend can be pinned to a schema.

Authentication is unchanged: main.py mounts this router behind the same API-key
dependency as every other router. The browser never holds that key — the
Cloudflare worker (see the frontend repo's worker/) is the only caller, and it
serves the browser from KV/R2 and its own cache.
"""
import datetime

from fastapi import APIRouter, HTTPException, status

import dependencies as dp
import web_contract as wc
from dependencies import Tags
from lib import timeit
from quantist_library import brokerflow as bf
from quantist_library import foreignflow as ff
from quantist_library import whaleflow as wf

# ==========
# Router Initiation
# ==========
router = APIRouter(
	prefix="/web-api/v1",
	tags=["web"],
	responses={404: {"description": status.HTTP_404_NOT_FOUND}}
)


# ==========
# Router
# ==========
# Both spellings reach the same handler: with no code at all the default
# instrument is COMPOSITE, which is what makes the web root a COMPOSITE page.
@router.get("/chart", status_code=status.HTTP_200_OK, response_model=wc.WebChart, tags=[Tags.web.name])
@router.get("/chart/{code}", status_code=status.HTTP_200_OK, response_model=wc.WebChart, tags=[Tags.web.name])
@timeit
async def get_web_chart(
	code: str = wc.DEFAULT_INSTRUMENT,
	method: dp.AnalysisMethod = dp.AnalysisMethod.broker,
	startdate: datetime.date | None = None,
	enddate: datetime.date = datetime.date.today(),
	clustering_method: dp.ClusteringMethod = dp.ClusteringMethod.correlation,
	) -> wc.WebChart:
	"""Semantic chart payload for one instrument. COMPOSITE and IHSG are the same instrument."""
	instrument = wc.resolve_instrument(code)

	try:
		if method == dp.AnalysisMethod.foreign:
			flow = wf.ForeignFlow(stockcode=instrument.code, startdate=startdate, enddate=enddate)
		else:
			flow = wf.BrokerFlow(stockcode=instrument.code, startdate=startdate, enddate=enddate, clustering_method=clustering_method)
		flow = await flow.fit()

	except KeyError as err:
		raise HTTPException(status.HTTP_404_NOT_FOUND, detail=err.args[0]) from err
	except ValueError as err:
		raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=err.args[0]) from err
	except Exception as err:
		raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=err.args[0]) from err

	return build_payload(flow, requested_code=code)


def build_payload(flow, requested_code: str | None = None) -> wc.WebChart:
	"""Read a fitted flow object into the contract. Kept separate so it stays testable."""
	bin_obj = getattr(flow, "bin_obj", None)
	return wc.build_web_chart(
		code=requested_code if requested_code else flow.stockcode,
		wf_indicators=flow.wf_indicators,
		analysis_method=flow.analysis_method.value,
		periods={
			"mf": flow.period_mf,
			"prop": flow.period_prop,
			"pricecorrel": flow.period_pricecorrel,
			"mapricecorrel": flow.period_mapricecorrel,
			"vwap": flow.period_vwap,
		},
		startdate=flow.startdate,
		enddate=flow.enddate,
		hist_bar=getattr(bin_obj, "hist_bar", None),
		peaks_index=getattr(bin_obj, "peaks_index", None),
		holding_composition=getattr(flow, "holding_composition", None),
		selected_brokers=getattr(flow, "selected_broker", None),
		optimum_n_selected_cluster=getattr(flow, "optimum_n_selected_cluster", None),
		optimum_corr=getattr(flow, "optimum_corr", None),
	)


# ==========
# Screener metadata
# ==========
@router.get("/screeners", status_code=status.HTTP_200_OK, response_model=wc.ScreenerCatalog, tags=[Tags.web.name])
@timeit
async def get_web_screeners() -> wc.ScreenerCatalog:
	"""
	Screener metadata, from the backend's own ScreenerList values.

	Each entry names its browser-facing results route and the legacy route
	Telegram still uses, and says whether the typed results can be fetched.
	The criteria come from ScreenerList, so this list cannot drift from what
	the backend actually supports.
	"""
	return wc.build_screener_catalog(slug.value for slug in dp.ScreenerList)


# ==========
# Screener results
# ==========
def _screener_object(slug: str, method: dp.AnalysisMethod, n_stockcodes: int, enddate: datetime.date):
	"""
	The screener class for one slug and method.

	Dispatch lives here rather than in the contract module: choosing a class is
	an API concern, and web_contract stays free of quantist_library imports so
	it can be tested without a database.
	"""
	family = wc.SCREENER_FAMILY[slug]
	library = ff if method == dp.AnalysisMethod.foreign else bf
	criterion = dp.ScreenerList(slug)

	# Each screener class narrows its criterion to the Literal subset it
	# handles. SCREENER_FAMILY is what guarantees the slug is in that subset,
	# and test_web_screener.py pins it against the enum, but the type checker
	# cannot follow the dict lookup — hence the ignores.
	if family == "money_flow":
		return library.ScreenerMoneyFlow(accum_or_distri=criterion, n_stockcodes=n_stockcodes, enddate=enddate)  # type: ignore[arg-type]
	if family == "vwap":
		return library.ScreenerVWAP(screener_vwap_criteria=criterion, n_stockcodes=n_stockcodes, enddate=enddate)  # type: ignore[arg-type]
	return library.ScreenerVProfile(screener_vprofile_criteria=criterion, n_stockcodes=n_stockcodes, enddate=enddate)  # type: ignore[arg-type]


@router.get("/screener/{slug}", status_code=status.HTTP_200_OK, response_model=wc.ScreenerResults, tags=[Tags.web.name])
@timeit
async def get_web_screener_results(
	slug: dp.ScreenerList,
	method: dp.AnalysisMethod = dp.AnalysisMethod.broker,
	n_stockcodes: int = 10,
	enddate: datetime.date = datetime.date.today(),
	) -> wc.ScreenerResults:
	"""
	Ranked results for one screener criterion, in one shape for every criterion.

	Additive: /whaleanalysis/screener/* keeps its DataFrame-dump response for
	Telegram and existing callers. This route normalizes the same objects into
	the semantic contract the browser reads, with the columns a criterion does
	not share carried through as extras rather than dropped.
	"""
	try:
		screener = _screener_object(slug.value, method, n_stockcodes, enddate)
		screener = await screener.screen()
		frame = screener.top_stockcodes

	except KeyError as err:
		raise HTTPException(status.HTTP_404_NOT_FOUND, detail=err.args[0]) from err
	except ValueError as err:
		raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=err.args[0]) from err
	except Exception as err:
		raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=err.args[0]) from err

	startdate = screener.startdate if isinstance(screener.startdate, datetime.date) else None
	return wc.build_screener_results(
		slug=slug.value,
		method=method.value,
		frame=frame,
		metadata={
			"startdate": startdate,
			"enddate": screener.enddate,
			"bar_range": getattr(screener, "bar_range", None) or getattr(screener, "radar_period", None),
		},
	)
