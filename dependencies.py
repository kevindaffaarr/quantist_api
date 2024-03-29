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
		from_attributes = True

class ListCode(BaseModel):
	index: int
	code: str
	value: Decimal | None = None
	frequency: Decimal | None = None
	foreignsellval: Decimal | None = None
	foreignbuyval: Decimal | None = None

	class Config:
		from_attributes = True

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

class ClusteringMethod(str, Enum):
	correlation = "correlation"
	timeseries = "timeseries"

class ScreenerList(str, Enum):
	most_accumulated = "most_accumulated"
	most_distributed = "most_distributed"
	vwap_rally = "vwap_rally"
	vwap_around = "vwap_around"
	vwap_breakout = "vwap_breakout"
	vwap_breakdown = "vwap_breakdown"
	vprofile_inside = "vprofile_inside"

class HoldingSectors(str, Enum):
	# Dictionary:
	# IS: Insurance
	# CP: Corporate
	# PF: Pension Fund
	# IB: Financial Instution (Investment Bank)
	# ID: Individual
	# MF: Mutual Fund
	# SC: Securities Company
	# FD: Foundation
	# OT: Others
	local_is = "local_is"
	local_cp = "local_cp"
	local_pf = "local_pf"
	local_ib = "local_ib"
	local_id = "local_id"
	local_mf = "local_mf"
	local_sc = "local_sc"
	local_fd = "local_fd"
	local_ot = "local_ot"
	local_total = "local_total"
	foreign_is = "foreign_is"
	foreign_cp = "foreign_cp"
	foreign_pf = "foreign_pf"
	foreign_ib = "foreign_ib"
	foreign_id = "foreign_id"
	foreign_mf = "foreign_mf"
	foreign_sc = "foreign_sc"
	foreign_fd = "foreign_fd"
	foreign_ot = "foreign_ot"
	foreign_total = "foreign_total"

class HoldingSectorsCat(dict, Enum):
	default = {
		"foreign": [HoldingSectors.foreign_total],
		"local_institutional": [
			HoldingSectors.local_is,
			HoldingSectors.local_cp,
			HoldingSectors.local_pf,
			HoldingSectors.local_ib,
			HoldingSectors.local_mf,
			HoldingSectors.local_sc,
			HoldingSectors.local_fd,
			HoldingSectors.local_ot,
		],
		"local_individual": [HoldingSectors.local_id]
	}