from fastapi import APIRouter, status, Depends
from fastapi.responses import Response
import datetime

import database as db
import dependencies as dp
from quantist_library import foreignflow as ff

# ==========
# Router Initiation
# ==========
router = APIRouter(
	prefix="/whaleanalysis",
	tags=["chart"],
	responses={404: {"description": status.HTTP_404_NOT_FOUND}}
)

# ==========
# Function
# ==========


# ==========
# Router
# ==========
@router.get("/")
def get_default_response():
	return """🔦 Quantist.io - Whale Analysis: Also called Transaction Analysis. The truly captured market action, whether someone buys or sells. Measuring the pressure of the market by Foreign Transactions, Broker Summary, Done Trade, Majority Shares Holder Transaction, and Shares Holder Distribution."""

@router.get("/foreign")
@router.get("/foreign/chart")
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
		chart = ff.StockFFFull(
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
			).chart(media_type=media_type)

	except KeyError as err:
		return err.args[0]

	else:
		if media_type in ["png","jpeg","jpg","webp"]:
			media_type = f"image/{media_type}"
		elif media_type == "svg":
			media_type = "image/svg+xml"
		else:
			media_type = f"application/json"
		return Response(content=chart, media_type=media_type)

@router.get("/foreign/full-data")
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
		return err.args[0]

	else:
		return full_data.to_dict(orient="index")

@router.get("/foreign/radar")
def get_foreign_radar():

	return 0