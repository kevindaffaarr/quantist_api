# ==========
# Pydantics Database Schema
# ==========
from unicodedata import numeric
from pydantic import BaseModel
from decimal import Decimal

class ListCodeBase(BaseModel):
	index: int
	code: str

	class Config:
		orm_mode = True

class ListCode(ListCodeBase):
	volume: Decimal | None = None
	frequency: Decimal | None = None
	foreignsell: Decimal | None = None
	foreignbuy: Decimal | None = None

	class Config:
		orm_mode = True

# ==========
# Parameter Class
# ==========
from enum import Enum
class ListCategory(str,Enum):
	broker = "broker"
	index = "index"
	stock = "stock"