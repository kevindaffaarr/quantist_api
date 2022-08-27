from fastapi import APIRouter, status, HTTPException, Query
from fastapi.responses import Response
import datetime

import dependencies as dp
from dependencies import Tags
from quantist_library import foreignflow as ff

# ==========
# Router Initiation
# ==========
router = APIRouter(
	prefix="/whaleanalysis",
	tags=["whaleanalysis"],
	responses={404: {"description": status.HTTP_404_NOT_FOUND}}
)

# ==========
# Function
# ==========


# ==========
# Router
# ==========
# DEFAULT ROUTER
# ==========
@router.get("/")
def get_default_response():
	return """🔦 Quantist.io - Whale Analysis: Also called Transaction Analysis. The truly captured market action, whether someone buys or sells. Measuring the pressure of the market by Foreign Transactions, Broker Summary, Done Trade, Majority Shares Holder Transaction, and Shares Holder Distribution."""

# ==========
# CHART ROUTER
# ==========
@router.get("/chart", tags=[Tags.chart.name])
@router.get("/chart/foreign", tags=[Tags.chart.name])
def get_foreign_chart(
	media_type:dp.ListMediaType | None = "json",
	stockcode: str | None = None,
	startdate: datetime.date | None = None,
	enddate: datetime.date | None = datetime.date.today(),
	period_fmf: int | None = None,
	period_fprop: int | None = None,
	period_fpricecorrel: int | None = None,
	period_fmapricecorrel: int | None = None,
	period_fvwap:int | None = None,
	fpow_high_fprop: int | None = None,
	fpow_high_fpricecorrel: int | None = None,
	fpow_high_fmapricecorrel: int | None = None,
	fpow_medium_fprop: int | None = None,
	fpow_medium_fpricecorrel: int | None = None,
	fpow_medium_fmapricecorrel: int | None = None,
):
	if media_type not in ["png","jpeg","jpg","webp","svg","json"]:
		media_type = "json"
	
	try:
		stock_ff_full = ff.StockFFFull(
			stockcode=stockcode,
			startdate=startdate,
			enddate=enddate,
			period_fmf=period_fmf,
			period_fprop=period_fprop,
			period_fpricecorrel=period_fpricecorrel,
			period_fmapricecorrel=period_fmapricecorrel,
			period_fvwap=period_fvwap,
			fpow_high_fprop=fpow_high_fprop,
			fpow_high_fpricecorrel=fpow_high_fpricecorrel,
			fpow_high_fmapricecorrel=fpow_high_fmapricecorrel,
			fpow_medium_fprop=fpow_medium_fprop,
			fpow_medium_fpricecorrel=fpow_medium_fpricecorrel,
			fpow_medium_fmapricecorrel=fpow_medium_fmapricecorrel,
			)
		
		chart = stock_ff_full.chart(media_type=media_type)

	except KeyError as err:
		raise HTTPException(status.HTTP_404_NOT_FOUND,detail=err.args[0])

	else:
		# Define the responses from Quantist: headers, content, and media_type
		# Define Headers
		headers = {
			"stockcode": stock_ff_full.stockcode,
			"last_date": stock_ff_full.ff_indicators.index[-1].strftime("%Y-%m-%d"),
			"last_fmf": stock_ff_full.ff_indicators['fmf'][-1].astype(str),
			"last_fprop": stock_ff_full.ff_indicators['fprop'][-1].astype(str),
			"last_fpricecorrel": stock_ff_full.ff_indicators['fpricecorrel'][-1].astype(str),
			"last_fmapricecorrel": stock_ff_full.ff_indicators['fmapricecorrel'][-1].astype(str),
			"last_fvwap": stock_ff_full.ff_indicators['fvwap'][-1].astype(str),
			"last_close": stock_ff_full.ff_indicators['close'][-1].astype(str),
		}
		
		# Define content
		# content=chart

		# Define media_type
		if media_type in ["png","jpeg","jpg","webp"]:
			media_type = f"image/{media_type}"
		elif media_type == "svg":
			media_type = "image/svg+xml"
		else:
			media_type = f"application/json"

		return Response(content=chart, media_type=media_type, headers=headers)

# ==========
# RADAR ROUTER
# ==========
@router.get("/radar", tags=[Tags.radar.name])
@router.get("/radar/foreign", tags=[Tags.radar.name])
def get_foreign_radar(
	media_type:dp.ListMediaType | None = "json",
	startdate: datetime.date | None = None,
	enddate: datetime.date | None = datetime.date.today(),
	y_axis_type: dp.ListRadarType | None = "correlation",
	stockcode_excludes: set[str] | None = Query(default=set()),
	include_composite: bool | None = False,
	screener_min_value: int | None = None,
	screener_min_frequency: int | None = None,
	screener_min_fprop:int | None = None,
):
	if media_type not in ["png","jpeg","jpg","webp","svg","json"]:
		media_type = "json"
	
	try:
		whale_radar_object = ff.WhaleRadar(
			startdate=startdate,
			enddate=enddate,
			y_axis_type=y_axis_type,
			stockcode_excludes=stockcode_excludes,
			include_composite=include_composite,
			screener_min_value=screener_min_value,
			screener_min_frequency=screener_min_frequency,
			screener_min_fprop=screener_min_fprop
		)
		
		chart = whale_radar_object.chart(media_type=media_type)

	except KeyError as err:
		raise HTTPException(status.HTTP_404_NOT_FOUND,detail=err.args[0])

	else:
		# Define the responses from Quantist: headers, content, and media_type
		# Define Headers
		headers = {
			"startdate": whale_radar_object.startdate.strftime("%Y-%m-%d"),
			"enddate": whale_radar_object.enddate.strftime("%Y-%m-%d"),
		}
		
		# Define content
		# content=chart

		# Define media_type
		if media_type in ["png","jpeg","jpg","webp"]:
			media_type = f"image/{media_type}"
		elif media_type == "svg":
			media_type = "image/svg+xml"
		else:
			media_type = f"application/json"
		
		return Response(content=chart, media_type=media_type, headers=headers)

# ==========
# FULL DATA ROUTER
# ==========
@router.get("/full-data", tags=[Tags.full_data.name])
@router.get("/full-data/foreign", tags=[Tags.full_data.name])
def get_foreign_data(
	stockcode: str | None = None,
	startdate: datetime.date | None = None,
	enddate: datetime.date | None = datetime.date.today(),
	period_fmf: int | None = None,
	period_fprop: int | None = None,
	period_fpricecorrel: int | None = None,
	period_fmapricecorrel: int | None = None,
	period_fvwap:int | None = None,
	fpow_high_fprop: int | None = None,
	fpow_high_fpricecorrel: int | None = None,
	fpow_high_fmapricecorrel: int | None = None,
	fpow_medium_fprop: int | None = None,
	fpow_medium_fpricecorrel: int | None = None,
	fpow_medium_fmapricecorrel: int | None = None,
):
	try:
		full_data = ff.StockFFFull(
			stockcode=stockcode,
			startdate=startdate,
			enddate=enddate,
			period_fmf=period_fmf,
			period_fprop=period_fprop,
			period_fpricecorrel=period_fpricecorrel,
			period_fmapricecorrel=period_fmapricecorrel,
			period_fvwap=period_fvwap,
			fpow_high_fprop=fpow_high_fprop,
			fpow_high_fpricecorrel=fpow_high_fpricecorrel,
			fpow_high_fmapricecorrel=fpow_high_fmapricecorrel,
			fpow_medium_fprop=fpow_medium_fprop,
			fpow_medium_fpricecorrel=fpow_medium_fpricecorrel,
			fpow_medium_fmapricecorrel=fpow_medium_fmapricecorrel,
			).ff_indicators
	
	except KeyError as err:
		raise HTTPException(status.HTTP_404_NOT_FOUND,detail=err.args[0])
	
	else:
		return full_data.to_dict(orient="index")

# ==========
# SCREENER ROUTER
# ==========
@router.get("/screener", tags=[Tags.screener.name])
@router.get("/screener/foreign", tags=[Tags.screener.name])
def get_screener_mostaccum():
	pass