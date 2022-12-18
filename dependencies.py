from pydantic import BaseModel
from enum import Enum
from decimal import Decimal

# ==========
# Pydantics Database Schema
# ==========
class DataParam(BaseModel):
	index: int
	param: str
	value: str

	class Config:
		orm_mode = True

class ListCode(BaseModel):
	index: int
	code: str
	value: Decimal | None = None
	frequency: Decimal | None = None
	foreignsellval: Decimal | None = None
	foreignbuyval: Decimal | None = None

	class Config:
		orm_mode = True

# ==========
# Metadata Tags
# ==========
class ExternalDocs(BaseModel):
	description:str | None
	url:str

class MetadataTag(BaseModel):
	name:str
	description:str | None
	external_docs:ExternalDocs | None

class Tags(Enum):
	chart = MetadataTag(
		name="chart",
		description="Generate chart by plotly, return plotly fig in json format as default, or png, jpeg, jpg, webp, and svg",
		external_docs=None
		)
	radar = MetadataTag(
		name="radar",
		description="Generate radar by plotly, return plotly fig in json format as default, or png, jpeg, jpg, webp, and svg",
		external_docs=None
		)
	screener = MetadataTag(
		name="screener",
		description="Find the list of stocks based on modifiable screener rules templates",
		external_docs=None
		)
	full_data = MetadataTag(
		name="full_data",
		description="Return full data of processed indicator from analysis in json format transformed from pandas dataframe",
		external_docs=None
		)

# ==========
# Parameter Class
# ==========
class ListCategory(str,Enum):
	broker = "broker"
	index = "index"  # type: ignore
	stock = "stock"

class ListMediaType(str,Enum):
	png = "png"
	jpeg = "jpeg"
	jpg = "jpg"
	webp = "webp"
	svg = "svg"
	json = "json"

class ListRadarType(str,Enum):
	correlation = "correlation"
	changepercentage = "changepercentage"

class ListBrokerApiType(str, Enum):
	all = "all"
	brokerflow = "brokerflow"
	brokercluster = "brokercluster"

class AnalysisMethod(str, Enum):
	foreign = "foreign"
	broker = "broker"

class ScreenerList(str, Enum):
	most_accumulated = "most_accumulated"
	most_distributed = "most_distributed"
	highprop_inflow = "highprop_inflow"
	highcorr_inflow = "highcorr_inflow"
	rebound_flow = "rebound_flow"
	drop_flow = "drop_flow"
	inflow_stayprice = "inflow_pricestay"
	neutral_downprice = "neutral_downprice"
	rally_flow = "rally_flow"
	crossup_vwap = "crossup_vwap"
	crossdown_vwap = "crossdown_vwap"
	price_around_vwap = "price_around_vwap"
	price_around_vwap_inflow = "price_around_vwap_inflow"
	peaking_transaction = "peaking_transaction"
