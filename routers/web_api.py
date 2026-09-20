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

from fastapi import APIRouter, Depends, HTTPException, status

import database as db
import dependencies as dp
import web_contract as wc
from dependencies import Tags
from lib import timeit
from quantist_library import whaleflow as wf
from routers.param import get_list_code

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
# Instrument catalogue and screener metadata
# ==========
@router.get("/instruments", status_code=status.HTTP_200_OK, response_model=wc.InstrumentList, tags=[Tags.web.name])
@timeit
async def get_web_instruments(
	category: dp.ListCategory = dp.ListCategory.stock,
	dbs: db.Session = Depends(db.get_dbs),
	) -> wc.InstrumentList:
	"""
	Typed instrument list for the web dropdown.

	Backed by the same table as /param/list/{category}, which keeps its own
	response shape for existing callers. This route adds the display
	resolution the dropdown needs — symbol, display name, index-vs-stock —
	through the same resolver the chart route uses, so COMPOSITE and IHSG
	cannot disagree between the list and the chart.
	"""
	try:
		rows = await get_list_code(dbs=dbs, list_category=category)
		# SQLAlchemy rows type as Column; the codes are strings on the wire.
		codes = [str(row.code) for row in rows] if isinstance(rows, list) else []
	except Exception as err:
		raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=err.args[0]) from err

	return wc.build_instrument_list(category=category.value, codes=codes)


@router.get("/screeners", status_code=status.HTTP_200_OK, response_model=wc.ScreenerCatalog, tags=[Tags.web.name])
@timeit
async def get_web_screeners() -> wc.ScreenerCatalog:
	"""
	Screener metadata, from the backend's own ScreenerList values.

	Metadata only, and it says so: every entry carries
	`web_results_available: false` and names the legacy endpoint that answers
	it today. Normalizing screener *results* into a typed web contract is the
	next step — see docs/eod-cache-contract.md. Until then a client can render
	the criteria honestly without implying it has the results.
	"""
	return wc.build_screener_catalog(slug.value for slug in dp.ScreenerList)
