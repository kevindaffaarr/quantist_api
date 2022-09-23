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
	description:str|None
	url:str

class MetadataTag(BaseModel):
	name:str
	description:str|None
	external_docs:ExternalDocs|None

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
