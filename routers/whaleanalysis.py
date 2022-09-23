from io import BytesIO
import datetime
import zipfile
from fastapi import APIRouter, status, HTTPException, Query
from fastapi.responses import Response, StreamingResponse

import dependencies as dp
from dependencies import Tags
from lib import timeit

from quantist_library import foreignflow as ff
from quantist_library import brokerflow as bf

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
@timeit
async def get_default_response():
	return """🔦 Quantist.io - Whale Analysis: Also called Transaction Analysis. The truly captured market action, whether someone buys or sells. Measuring the pressure of the market by Foreign Transactions, Broker Summary, Done Trade, Majority Shares Holder Transaction, and Shares Holder Distribution."""

# ==========
# CHART ROUTER
# ==========
@router.get("/chart", tags=[Tags.chart.name])
@router.get("/chart/foreign", tags=[Tags.chart.name])
@timeit
async def get_foreign_chart(
	media_type:dp.ListMediaType | None = dp.ListMediaType.json,
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
	if media_type not in [dp.ListMediaType.png,
		dp.ListMediaType.jpeg,
		dp.ListMediaType.jpg,
		dp.ListMediaType.webp,
		dp.ListMediaType.svg,
		dp.ListMediaType.json
		]:
		media_type = dp.ListMediaType.json
	
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
		stock_ff_full = await stock_ff_full.fit()
		chart = await stock_ff_full.chart(media_type=media_type)

	except KeyError as err:
		raise HTTPException(status.HTTP_404_NOT_FOUND,detail=err.args[0]) from err

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
		if media_type in [dp.ListMediaType.png,
			dp.ListMediaType.jpeg,
			dp.ListMediaType.jpg,
			dp.ListMediaType.webp
			]:
			media_type = f"image/{media_type}"
		elif media_type == dp.ListMediaType.svg:
			media_type = "image/svg+xml"
		else:
			media_type = "application/json"

		return Response(content=chart, media_type=media_type, headers=headers)

@router.get("/chart/broker", tags=[Tags.chart.name])
@timeit
async def get_broker_chart(
	media_type:dp.ListMediaType | None = dp.ListMediaType.json,
	api_type: dp.ListBrokerApiType | None = dp.ListBrokerApiType.brokerflow,
	stockcode: str | None = None,
	startdate: datetime.date | None = None,
	enddate: datetime.date | None = datetime.date.today(),
	n_selected_cluster:int | None = None,
	period_wmf: int | None = None,
	period_wprop: int | None = None,
	period_wpricecorrel: int | None = None,
	period_wmapricecorrel: int | None = None,
	period_wvwap:int | None = None,
	wpow_high_wprop: int | None = None,
	wpow_high_wpricecorrel: int | None = None,
	wpow_high_wmapricecorrel: int | None = None,
	wpow_medium_wprop: int | None = None,
	wpow_medium_wpricecorrel: int | None = None,
	wpow_medium_wmapricecorrel: int | None = None,
	training_start_index: int | None = None,
	training_end_index: int | None = None,
	min_n_cluster: int | None = None,
	max_n_cluster: int | None = None,
	splitted_min_n_cluster: int | None = None,
	splitted_max_n_cluster: int | None = None,
	stepup_n_cluster_threshold: int | None = None,
):
	if media_type not in [dp.ListMediaType.png,
		dp.ListMediaType.jpeg,
		dp.ListMediaType.jpg,
		dp.ListMediaType.webp,
		dp.ListMediaType.svg,
		dp.ListMediaType.json
		]:
		media_type = dp.ListMediaType.json
	
	try:
		stock_bf_full = bf.StockBFFull(
			stockcode=stockcode,
			startdate=startdate,
			enddate=enddate,
			n_selected_cluster=n_selected_cluster,
			period_wmf=period_wmf,
			period_wprop=period_wprop,
			period_wpricecorrel=period_wpricecorrel,
			period_wmapricecorrel=period_wmapricecorrel,
			period_wvwap=period_wvwap,
			wpow_high_wprop=wpow_high_wprop,
			wpow_high_wpricecorrel=wpow_high_wpricecorrel,
			wpow_high_wmapricecorrel=wpow_high_wmapricecorrel,
			wpow_medium_wprop=wpow_medium_wprop,
			wpow_medium_wpricecorrel=wpow_medium_wpricecorrel,
			wpow_medium_wmapricecorrel=wpow_medium_wmapricecorrel,
			training_start_index=training_start_index,
			training_end_index=training_end_index,
			min_n_cluster=min_n_cluster,
			max_n_cluster=max_n_cluster,
			splitted_min_n_cluster=splitted_min_n_cluster,
			splitted_max_n_cluster=splitted_max_n_cluster,
			stepup_n_cluster_threshold=stepup_n_cluster_threshold,
			)
		stock_bf_full = await stock_bf_full.fit()
		if api_type == dp.ListBrokerApiType.brokerflow:
			chart = await stock_bf_full.chart(media_type=media_type)
		elif api_type == dp.ListBrokerApiType.brokercluster:
			chart = await stock_bf_full.broker_cluster_chart(media_type=media_type)
		elif api_type == dp.ListBrokerApiType.all:
			chart_flow = await stock_bf_full.chart(media_type=media_type)
			chart_cluster = await stock_bf_full.broker_cluster_chart(media_type=media_type)
			# Create zip file to be sent through API
			chart_all = {
				"flow": chart_flow,
				"cluster": chart_cluster,
			}
			chart = BytesIO()
			with zipfile.ZipFile(chart, "a", zipfile.ZIP_DEFLATED, False) as zf:
				for key, value in chart_all.items():
					zf.writestr(f"{key}.{media_type}", value)
			chart.seek(0)
		else:
			raise ValueError("api_type is not valid")

	except KeyError as err:
		raise HTTPException(status.HTTP_404_NOT_FOUND,detail=err.args[0]) from err
	except ValueError as err:
		raise HTTPException(status.HTTP_400_BAD_REQUEST,detail=err.args[0]) from err

	else:
		# Define the responses from Quantist: headers, content, and media_type
		# Define Headers
		headers = {
			"stockcode": stock_bf_full.stockcode,
			"last_date": stock_bf_full.bf_indicators.index[-1].strftime("%Y-%m-%d"),
			"last_wmf": stock_bf_full.bf_indicators['wmf'][-1].astype(str),
			"last_wprop": stock_bf_full.bf_indicators['wprop'][-1].astype(str),
			"last_wpricecorrel": stock_bf_full.bf_indicators['wpricecorrel'][-1].astype(str),
			"last_wmapricecorrel": stock_bf_full.bf_indicators['wmapricecorrel'][-1].astype(str),
			"last_wvwap": stock_bf_full.bf_indicators['wvwap'][-1].astype(str),
			"last_close": stock_bf_full.bf_indicators['close'][-1].astype(str),
		}

		# Define content
		# content=chart

		# Define media_type
		if api_type == dp.ListBrokerApiType.all:
			media_type = "application/zip"
			return StreamingResponse(chart, media_type=media_type, headers=headers)

		else:
			if media_type in [dp.ListMediaType.png,
				dp.ListMediaType.jpeg,
				dp.ListMediaType.jpg,
				dp.ListMediaType.webp
				]:
				media_type = f"image/{media_type}"
			elif media_type == dp.ListMediaType.svg:
				media_type = "image/svg+xml"
			else:
				media_type = "application/json"
			return Response(content=chart, media_type=media_type, headers=headers)
			

# ==========
# RADAR ROUTER
# ==========
@router.get("/radar", tags=[Tags.radar.name])
@router.get("/radar/foreign", tags=[Tags.radar.name])
@timeit
async def get_foreign_radar(
	media_type:dp.ListMediaType | None = dp.ListMediaType.json,
	startdate: datetime.date | None = None,
	enddate: datetime.date | None = datetime.date.today(),
	y_axis_type: dp.ListRadarType | None = "correlation",
	stockcode_excludes: set[str] | None = Query(default=set()),
	include_composite: bool | None = False,
	screener_min_value: int | None = None,
	screener_min_frequency: int | None = None,
	screener_min_fprop:int | None = None,
):
	if media_type not in [dp.ListMediaType.png,
		dp.ListMediaType.jpeg,
		dp.ListMediaType.jpg,
		dp.ListMediaType.webp,
		dp.ListMediaType.svg,
		dp.ListMediaType.json
		]:
		media_type = dp.ListMediaType.json
	
	try:
		whale_radar_object = ff.ForeignRadar(
			startdate=startdate,
			enddate=enddate,
			y_axis_type=y_axis_type,
			stockcode_excludes=stockcode_excludes,
			include_composite=include_composite,
			screener_min_value=screener_min_value,
			screener_min_frequency=screener_min_frequency,
			screener_min_fprop=screener_min_fprop
		)
		whale_radar_object = await whale_radar_object.fit()
		
		chart = await whale_radar_object.chart(media_type=media_type)

	except KeyError as err:
		raise HTTPException(status.HTTP_404_NOT_FOUND,detail=err.args[0]) from err

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
		if media_type in [dp.ListMediaType.png,
			dp.ListMediaType.jpeg,
			dp.ListMediaType.jpg,
			dp.ListMediaType.webp
			]:
			media_type = f"image/{media_type}"
		elif media_type == dp.ListMediaType.svg:
			media_type = "image/svg+xml"
		else:
			media_type = "application/json"
		
		return Response(content=chart, media_type=media_type, headers=headers)

# ==========
# FULL DATA ROUTER
# ==========
@router.get("/full-data", tags=[Tags.full_data.name])
@router.get("/full-data/foreign", tags=[Tags.full_data.name])
@timeit
async def get_foreign_data(
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
		stock_ff_full = await stock_ff_full.fit()
		full_data = stock_ff_full.ff_indicators
	
	except KeyError as err:
		raise HTTPException(status.HTTP_404_NOT_FOUND,detail=err.args[0]) from err
	
	else:
		return full_data.to_dict(orient="index")

# ==========
# SCREENER ROUTER
# ==========
@router.get("/screener", tags=[Tags.screener.name])
@router.get("/screener/foreign", tags=[Tags.screener.name])
@timeit
async def get_screener_mostaccum():
	pass
