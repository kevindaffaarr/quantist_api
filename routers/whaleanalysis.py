from typing import Literal
from io import BytesIO
import datetime
import zipfile
import pandas as pd
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
	media_type:dp.ListMediaType = dp.ListMediaType.json,
	stockcode: str | None = None,
	startdate: datetime.date | None = None,
	enddate: datetime.date = datetime.date.today(),
	period_mf: int | None = None,
	period_prop: int | None = None,
	period_pricecorrel: int | None = None,
	period_mapricecorrel: int | None = None,
	period_vwap:int | None = None,
	pow_high_prop: int | None = None,
	pow_high_pricecorrel: int | None = None,
	pow_high_mapricecorrel: int | None = None,
	pow_medium_prop: int | None = None,
	pow_medium_pricecorrel: int | None = None,
	pow_medium_mapricecorrel: int | None = None,
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
			period_mf=period_mf,
			period_prop=period_prop,
			period_pricecorrel=period_pricecorrel,
			period_mapricecorrel=period_mapricecorrel,
			period_vwap=period_vwap,
			pow_high_prop=pow_high_prop,
			pow_high_pricecorrel=pow_high_pricecorrel,
			pow_high_mapricecorrel=pow_high_mapricecorrel,
			pow_medium_prop=pow_medium_prop,
			pow_medium_pricecorrel=pow_medium_pricecorrel,
			pow_medium_mapricecorrel=pow_medium_mapricecorrel,
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
			"last_date": stock_ff_full.ff_indicators.index[-1].strftime("%Y-%m-%d"), # type: ignore
			"last_mf": stock_ff_full.ff_indicators['mf'][-1].astype(str),
			"last_prop": stock_ff_full.ff_indicators['prop'][-1].astype(str),
			"last_pricecorrel": stock_ff_full.ff_indicators['pricecorrel'][-1].astype(str),
			"last_mapricecorrel": stock_ff_full.ff_indicators['mapricecorrel'][-1].astype(str),
			"last_vwap": stock_ff_full.ff_indicators['vwap'][-1].astype(str),
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
			mime_type:str = f"image/{media_type}"
		elif media_type == dp.ListMediaType.svg:
			mime_type:str = "image/svg+xml"
		else:
			mime_type:str = "application/json"

		return Response(content=chart, media_type=mime_type, headers=headers)

@router.get("/chart/broker", tags=[Tags.chart.name])
@timeit
async def get_broker_chart(
	media_type:dp.ListMediaType = dp.ListMediaType.json,
	api_type: dp.ListBrokerApiType = dp.ListBrokerApiType.brokerflow,
	stockcode: str | None = None,
	startdate: datetime.date | None = None,
	enddate: datetime.date = datetime.date.today(),
	n_selected_cluster:int | None = None,
	period_mf: int | None = None,
	period_prop: int | None = None,
	period_pricecorrel: int | None = None,
	period_mapricecorrel: int | None = None,
	period_vwap:int | None = None,
	pow_high_prop: int | None = None,
	pow_high_pricecorrel: int | None = None,
	pow_high_mapricecorrel: int | None = None,
	pow_medium_prop: int | None = None,
	pow_medium_pricecorrel: int | None = None,
	pow_medium_mapricecorrel: int | None = None,
	training_start_index: float | None = None,
	training_end_index: float | None = None,
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
			period_mf=period_mf,
			period_prop=period_prop,
			period_pricecorrel=period_pricecorrel,
			period_mapricecorrel=period_mapricecorrel,
			period_vwap=period_vwap,
			pow_high_prop=pow_high_prop,
			pow_high_pricecorrel=pow_high_pricecorrel,
			pow_high_mapricecorrel=pow_high_mapricecorrel,
			pow_medium_prop=pow_medium_prop,
			pow_medium_pricecorrel=pow_medium_pricecorrel,
			pow_medium_mapricecorrel=pow_medium_mapricecorrel,
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
					zf.writestr(f"{key}.{media_type}", value) # type: ignore
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
			"last_date": stock_bf_full.bf_indicators.index[-1].strftime("%Y-%m-%d"), # type: ignore
			"last_mf": stock_bf_full.bf_indicators['mf'][-1].astype(str),
			"last_prop": stock_bf_full.bf_indicators['prop'][-1].astype(str),
			"last_pricecorrel": stock_bf_full.bf_indicators['pricecorrel'][-1].astype(str),
			"last_mapricecorrel": stock_bf_full.bf_indicators['mapricecorrel'][-1].astype(str),
			"last_vwap": stock_bf_full.bf_indicators['vwap'][-1].astype(str),
			"last_close": stock_bf_full.bf_indicators['close'][-1].astype(str),
		}

		# Define content
		# content=chart

		# Define media_type
		if api_type == dp.ListBrokerApiType.all:
			mimetype:str = "application/zip"
			return StreamingResponse(chart, media_type=media_type, headers=headers) # type: ignore

		else:
			if media_type in [dp.ListMediaType.png,
				dp.ListMediaType.jpeg,
				dp.ListMediaType.jpg,
				dp.ListMediaType.webp
				]:
				mimetype:str = f"image/{media_type}"
			elif media_type == dp.ListMediaType.svg:
				mimetype:str = "image/svg+xml"
			else:
				mimetype:str = "application/json"
			return Response(content=chart, media_type=mimetype, headers=headers)
			

# ==========
# RADAR ROUTER
# ==========
@router.get("/radar", tags=[Tags.radar.name])
@router.get("/radar/foreign", tags=[Tags.radar.name])
@timeit
async def get_foreign_radar(
	media_type:dp.ListMediaType = dp.ListMediaType.json,
	startdate: datetime.date | None = None,
	enddate: datetime.date = datetime.date.today(),
	y_axis_type: dp.ListRadarType = dp.ListRadarType.correlation,
	stockcode_excludes: set[str] = Query(default=set()),
	include_composite: bool = False,
	screener_min_value: int | None = None,
	screener_min_frequency: int | None = None,
	screener_min_prop:int | None = None,
	period_mf: int | None = None,
	period_pricecorrel: int | None = None,
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
			screener_min_prop=screener_min_prop,
			period_mf=period_mf,
			period_pricecorrel=period_pricecorrel,
		)
		whale_radar_object = await whale_radar_object.fit()
		
		chart = await whale_radar_object.chart(media_type=media_type)

	except KeyError as err:
		raise HTTPException(status.HTTP_404_NOT_FOUND,detail=err.args[0]) from err

	else:
		# Define the responses from Quantist: headers, content, and media_type
		# Define Headers
		assert whale_radar_object.startdate is not None
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
			mime_type:str = f"image/{media_type}"
		elif media_type == dp.ListMediaType.svg:
			mime_type:str = "image/svg+xml"
		else:
			mime_type:str = "application/json"
		
		return Response(content=chart, media_type=mime_type, headers=headers)

@router.get("/radar/broker", tags=[Tags.radar.name])
@timeit
async def get_broker_radar(
	media_type:dp.ListMediaType = dp.ListMediaType.json,
	startdate: datetime.date | None = None,
	enddate: datetime.date = datetime.date.today(),
	y_axis_type: dp.ListRadarType = dp.ListRadarType.correlation,
	stockcode_excludes: set[str] = Query(default=set()),
	include_composite: bool = False,
	screener_min_value: int | None = None,
	screener_min_frequency: int | None = None,
	n_selected_cluster:int | None = None,
	radar_period: int | None = None,
	period_mf: int | None = None,
	period_pricecorrel: int | None = None,
	default_months_range: int | None = None,
	training_start_index: float | None = None,
	training_end_index: float | None = None,
	min_n_cluster: int | None = None,
	max_n_cluster: int | None = None,
	splitted_min_n_cluster: int | None = None,
	splitted_max_n_cluster: int | None = None,
	stepup_n_cluster_threshold: int | None = None,
	filter_opt_corr: int | None = None,
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
		whale_radar_object = bf.WhaleRadar(
			startdate=startdate,
			enddate=enddate,
			y_axis_type=y_axis_type,
			stockcode_excludes=stockcode_excludes,
			include_composite=include_composite,
			screener_min_value=screener_min_value,
			screener_min_frequency=screener_min_frequency,
			n_selected_cluster=n_selected_cluster,
			radar_period=radar_period,
			period_mf=period_mf,
			period_pricecorrel=period_pricecorrel,
			default_months_range=default_months_range,
			training_start_index=training_start_index,
			training_end_index=training_end_index,
			min_n_cluster=min_n_cluster,
			max_n_cluster=max_n_cluster,
			splitted_min_n_cluster=splitted_min_n_cluster,
			splitted_max_n_cluster=splitted_max_n_cluster,
			stepup_n_cluster_threshold=stepup_n_cluster_threshold,
			filter_opt_corr=filter_opt_corr,
		)
		whale_radar_object = await whale_radar_object.fit()
		
		chart = await whale_radar_object.chart(media_type=media_type)

	except KeyError as err:
		raise HTTPException(status.HTTP_404_NOT_FOUND,detail=err.args[0]) from err

	else:
		# Define the responses from Quantist: headers, content, and media_type
		# Define Headers
		headers = {
			"startdate": whale_radar_object.startdate.strftime("%Y-%m-%d"), # type: ignore
			"enddate": whale_radar_object.enddate.strftime("%Y-%m-%d"), # type: ignore
		}
		
		# Define content
		# content=chart

		# Define media_type
		if media_type in [dp.ListMediaType.png,
			dp.ListMediaType.jpeg,
			dp.ListMediaType.jpg,
			dp.ListMediaType.webp
			]:
			mime_type:str = f"image/{media_type}"
		elif media_type == dp.ListMediaType.svg:
			mime_type:str = "image/svg+xml"
		else:
			mime_type:str = "application/json"
		
		return Response(content=chart, media_type=mime_type, headers=headers)


# ==========
# FULL DATA ROUTER
# ==========
@router.get("/full-data", tags=[Tags.full_data.name])
@router.get("/full-data/foreign", tags=[Tags.full_data.name])
@timeit
async def get_foreign_data(
	stockcode: str | None = None,
	startdate: datetime.date | None = None,
	enddate: datetime.date = datetime.date.today(),
	period_mf: int | None = None,
	period_prop: int | None = None,
	period_pricecorrel: int | None = None,
	period_mapricecorrel: int | None = None,
	period_vwap:int | None = None,
	pow_high_prop: int | None = None,
	pow_high_pricecorrel: int | None = None,
	pow_high_mapricecorrel: int | None = None,
	pow_medium_prop: int | None = None,
	pow_medium_pricecorrel: int | None = None,
	pow_medium_mapricecorrel: int | None = None,
):
	try:
		stock_ff_full = ff.StockFFFull(
			stockcode=stockcode,
			startdate=startdate,
			enddate=enddate,
			period_mf=period_mf,
			period_prop=period_prop,
			period_pricecorrel=period_pricecorrel,
			period_mapricecorrel=period_mapricecorrel,
			period_vwap=period_vwap,
			pow_high_prop=pow_high_prop,
			pow_high_pricecorrel=pow_high_pricecorrel,
			pow_high_mapricecorrel=pow_high_mapricecorrel,
			pow_medium_prop=pow_medium_prop,
			pow_medium_pricecorrel=pow_medium_pricecorrel,
			pow_medium_mapricecorrel=pow_medium_mapricecorrel,
			)
		stock_ff_full = await stock_ff_full.fit()
		full_data = stock_ff_full.ff_indicators
	
	except KeyError as err:
		raise HTTPException(status.HTTP_404_NOT_FOUND,detail=err.args[0]) from err
	
	else:
		return full_data.to_dict(orient="index")

@router.get("/full-data/broker", tags=[Tags.full_data.name])
@timeit
async def get_broker_data(
	api_type: dp.ListBrokerApiType = dp.ListBrokerApiType.brokerflow,
	stockcode: str | None = None,
	startdate: datetime.date | None = None,
	enddate: datetime.date = datetime.date.today(),
	n_selected_cluster:int | None = None,
	period_mf: int | None = None,
	period_prop: int | None = None,
	period_pricecorrel: int | None = None,
	period_mapricecorrel: int | None = None,
	period_vwap:int | None = None,
	pow_high_prop: int | None = None,
	pow_high_pricecorrel: int | None = None,
	pow_high_mapricecorrel: int | None = None,
	pow_medium_prop: int | None = None,
	pow_medium_pricecorrel: int | None = None,
	pow_medium_mapricecorrel: int | None = None,
	training_start_index: float | None = None,
	training_end_index: float | None = None,
	min_n_cluster: int | None = None,
	max_n_cluster: int | None = None,
	splitted_min_n_cluster: int | None = None,
	splitted_max_n_cluster: int | None = None,
	stepup_n_cluster_threshold: int | None = None,
):
	try:
		stock_bf_full = bf.StockBFFull(
			stockcode=stockcode,
			startdate=startdate,
			enddate=enddate,
			n_selected_cluster=n_selected_cluster,
			period_mf=period_mf,
			period_prop=period_prop,
			period_pricecorrel=period_pricecorrel,
			period_mapricecorrel=period_mapricecorrel,
			period_vwap=period_vwap,
			pow_high_prop=pow_high_prop,
			pow_high_pricecorrel=pow_high_pricecorrel,
			pow_high_mapricecorrel=pow_high_mapricecorrel,
			pow_medium_prop=pow_medium_prop,
			pow_medium_pricecorrel=pow_medium_pricecorrel,
			pow_medium_mapricecorrel=pow_medium_mapricecorrel,
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
			data = stock_bf_full.bf_indicators.to_dict(orient="index"),
		elif api_type == dp.ListBrokerApiType.brokercluster:
			data = stock_bf_full.broker_features.to_dict(orient="index"),
		elif api_type == dp.ListBrokerApiType.all:
			flow = stock_bf_full.bf_indicators
			cluster = stock_bf_full.broker_features
			data = {
				"flow": flow.to_dict(orient="index"),
				"cluster": cluster.to_dict(orient="index"),
			}
		else:
			raise ValueError("api_type is not valid")
	
	except KeyError as err:
		raise HTTPException(status.HTTP_404_NOT_FOUND,detail=err.args[0]) from err
	
	else:
		return data


# ==========
# SCREENER ROUTER
# ==========
@router.get("/screener", tags=[Tags.screener.name])
@router.get("/screener/foreign", tags=[Tags.screener.name])
@router.get("/screener/foreign/top-money-flow", tags=[Tags.screener.name])
@timeit
async def get_screener_foreign_moneyflow(
	accum_or_distri: Literal[dp.ScreenerList.most_accumulated,dp.ScreenerList.most_distributed] = dp.ScreenerList.most_accumulated,
	n_stockcodes: int = 10,
	startdate: datetime.date | None = None,
	enddate: datetime.date = datetime.date.today(),
	screener_period: int | None = None,
	stockcode_excludes: set[str] = Query(default=set()),
	screener_min_value: int | None = None,
	screener_min_frequency: int | None = None,
	screener_min_prop:int | None = None,
	):
	try:
		screener_moneyflow_object = ff.ScreenerMoneyFlow(
			accum_or_distri=accum_or_distri,
			n_stockcodes=n_stockcodes,
			startdate=startdate,
			enddate=enddate,
			screener_period=screener_period,
			stockcode_excludes=stockcode_excludes,
			screener_min_value=screener_min_value,
			screener_min_frequency=screener_min_frequency,
			screener_min_prop=screener_min_prop,
		)
		screener_moneyflow_object = await screener_moneyflow_object.screen()

		top_stockcodes = screener_moneyflow_object.top_stockcodes
	
	except Exception as err:
		raise HTTPException(status.HTTP_400_BAD_REQUEST,detail=err.args[0]) from err
	
	else:
		# Define screener_metadata
		screener_metadata = {
			"analysis_method": dp.AnalysisMethod.foreign,
			"screener_method": accum_or_distri,
			"bar_range": screener_moneyflow_object.bar_range,
			"enddate": screener_moneyflow_object.enddate.strftime("%Y-%m-%d"), # type: ignore
		}
		if isinstance(screener_moneyflow_object.startdate, datetime.date):
			screener_metadata["startdate"] = screener_moneyflow_object.startdate.strftime("%Y-%m-%d")
		else:
			screener_metadata["startdate"] = None

	# Return screener_metadata and top_stockcodes
	return {
		"screener_metadata":screener_metadata,
		"top_stockcodes":top_stockcodes.to_dict(orient="index")
		}

@router.get("/screener/broker/top-money-flow", tags=[Tags.screener.name])
@timeit
async def get_screener_broker_moneyflow(
	accum_or_distri: Literal[dp.ScreenerList.most_accumulated,dp.ScreenerList.most_distributed] = dp.ScreenerList.most_accumulated,
	n_stockcodes: int = 10,
	startdate: datetime.date | None = None,
	enddate: datetime.date = datetime.date.today(),
	stockcode_excludes: set[str] = Query(default=set()),
	screener_min_value: int | None = None,
	screener_min_frequency: int | None = None,
	n_selected_cluster:int | None = None,
	radar_period: int | None = None,
	period_mf: int | None = None,
	period_pricecorrel: int | None = None,
	default_months_range: int | None = None,
	training_start_index: float | None = None,
	training_end_index: float | None = None,
	min_n_cluster: int | None = None,
	max_n_cluster: int | None = None,
	splitted_min_n_cluster: int | None = None,
	splitted_max_n_cluster: int | None = None,
	stepup_n_cluster_threshold: int | None = None,
	filter_opt_corr: int | None = None,
	):
	try:
		screener_moneyflow_object = bf.ScreenerMoneyFlow(
			accum_or_distri = accum_or_distri,
			n_stockcodes = n_stockcodes,
			startdate = startdate,
			enddate = enddate,
			stockcode_excludes = stockcode_excludes,
			screener_min_value = screener_min_value,
			screener_min_frequency = screener_min_frequency,
			n_selected_cluster = n_selected_cluster,
			radar_period = radar_period,
			period_mf = period_mf,
			period_pricecorrel = period_pricecorrel,
			default_months_range = default_months_range,
			training_start_index = training_start_index,
			training_end_index = training_end_index,
			min_n_cluster = min_n_cluster,
			max_n_cluster = max_n_cluster,
			splitted_min_n_cluster = splitted_min_n_cluster,
			splitted_max_n_cluster = splitted_max_n_cluster,
			stepup_n_cluster_threshold = stepup_n_cluster_threshold,
			filter_opt_corr = filter_opt_corr,
		)
		screener_moneyflow_object = await screener_moneyflow_object.screen()
		top_stockcodes = screener_moneyflow_object.top_stockcodes
	
	except Exception as err:
		raise HTTPException(status.HTTP_400_BAD_REQUEST,detail=err.args[0]) from err
	
	else:
		# Define screener_metadata
		screener_metadata = {
			"analysis_method": dp.AnalysisMethod.broker,
			"screener_method": accum_or_distri,
			"bar_range": screener_moneyflow_object.radar_period,
			"enddate": screener_moneyflow_object.enddate.strftime("%Y-%m-%d"), # type: ignore
		}
		if isinstance(screener_moneyflow_object.startdate, datetime.date):
			screener_metadata["startdate"] = screener_moneyflow_object.startdate.strftime("%Y-%m-%d")
		else:
			screener_metadata["startdate"] = None

	# Return screener_metadata and top_stockcodes
	return {
		"screener_metadata":screener_metadata,
		"top_stockcodes":top_stockcodes.to_dict(orient="index")
		}