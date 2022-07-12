# ==========
# Pydantics Database Schema
# ==========
from pydantic import BaseModel
from decimal import Decimal

class ListStockBase(BaseModel):
	code: str

class ListStockExtended(ListStockBase):
	index: int
	volume: Decimal | None = None
	frequency: Decimal | None = None
	foreignsell: Decimal | None = None
	foreignbuy: Decimal | None = None

	class Config:
		orm_mode = True

class ListStock(ListStockBase):
	index: int

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