from __future__ import annotations
from fastapi_globals import g

import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
from sqlalchemy.sql import func

import database as db
import dependencies as dp

pd.options.mode.copy_on_write = True
pd.options.future.infer_string = True

class HoldingComposition():
	"""
	Input: stockcode, date range, categorization
	Output: Holding Composition by percentage of each category for every date range (monthly)
	Process:
		> Date validation: >= 2015-01-01 and <= last month
		> Validation does the stockcode exists
		> Get the data from database filtered by stockcode, sectype, and date range
		> Group by date and sum the value of each category
		> Calculate the percentage of each category for every date
		> Return the result
	"""
	def __init__(self,
		stockcode: str,
		startdate: datetime.date|None = None,
		enddate: datetime.date|None = datetime.date.today().replace(day=1) - datetime.timedelta(days=1),
		categorization: dp.HoldingSectorsCat|None = dp.HoldingSectorsCat.default,
		) -> None:
		"""
		Class initiation, and validate startdate and enddate
		"""
		assert isinstance(enddate, datetime.date)
		assert isinstance(categorization, dp.HoldingSectorsCat)

		self.stockcode: str = stockcode.lower()
		self.startdate: datetime.date | None = startdate
		self.enddate: datetime.date = enddate
		self.categorization: dp.HoldingSectorsCat = categorization

		# Validation startdate and enddate
		if isinstance(self.startdate, datetime.date) and self.startdate < datetime.date(2015, 1, 1):
			raise ValueError("Startdate must be >= 2015-01-01")
		if self.enddate > datetime.date.today().replace(day=1) - datetime.timedelta(days=1):
			raise ValueError("Enddate must be <= last month")
	
	async def __get_param(self, dbs:db.Session = next(db.get_dbs())) -> pd.Series:
		# Check does g.DEFAULT_PARAM is available and is a pandas series
		if "g" in globals() and hasattr(g, "DEFAULT_PARAM") and isinstance(g.DEFAULT_PARAM, pd.Series):
			default_param = g.DEFAULT_PARAM
		else:
			default_param = await db.get_default_param()
		return default_param

	async def __get_data_ksei(self, dbs:db.Session = next(db.get_dbs())) -> pd.DataFrame:
		"""
		Get the data from database filtered by stockcode, sectype, and date range
		"""
		# Get the data from database
		qry = dbs.query(
				db.KseiKepemilikanEfek.date,
				func.sum(db.KseiKepemilikanEfek.price * db.KseiKepemilikanEfek.local_is).label('local_is'),
				func.sum(db.KseiKepemilikanEfek.price * db.KseiKepemilikanEfek.local_cp).label('local_cp'),
				func.sum(db.KseiKepemilikanEfek.price * db.KseiKepemilikanEfek.local_pf).label('local_pf'),
				func.sum(db.KseiKepemilikanEfek.price * db.KseiKepemilikanEfek.local_ib).label('local_ib'),
				func.sum(db.KseiKepemilikanEfek.price * db.KseiKepemilikanEfek.local_id).label('local_id'),
				func.sum(db.KseiKepemilikanEfek.price * db.KseiKepemilikanEfek.local_mf).label('local_mf'),
				func.sum(db.KseiKepemilikanEfek.price * db.KseiKepemilikanEfek.local_sc).label('local_sc'),
				func.sum(db.KseiKepemilikanEfek.price * db.KseiKepemilikanEfek.local_fd).label('local_fd'),
				func.sum(db.KseiKepemilikanEfek.price * db.KseiKepemilikanEfek.local_ot).label('local_ot'),
				func.sum(db.KseiKepemilikanEfek.price * db.KseiKepemilikanEfek.local_total).label('local_total'),
				func.sum(db.KseiKepemilikanEfek.price * db.KseiKepemilikanEfek.foreign_is).label('foreign_is'),
				func.sum(db.KseiKepemilikanEfek.price * db.KseiKepemilikanEfek.foreign_cp).label('foreign_cp'),
				func.sum(db.KseiKepemilikanEfek.price * db.KseiKepemilikanEfek.foreign_pf).label('foreign_pf'),
				func.sum(db.KseiKepemilikanEfek.price * db.KseiKepemilikanEfek.foreign_ib).label('foreign_ib'),
				func.sum(db.KseiKepemilikanEfek.price * db.KseiKepemilikanEfek.foreign_id).label('foreign_id'),
				func.sum(db.KseiKepemilikanEfek.price * db.KseiKepemilikanEfek.foreign_mf).label('foreign_mf'),
				func.sum(db.KseiKepemilikanEfek.price * db.KseiKepemilikanEfek.foreign_sc).label('foreign_sc'),
				func.sum(db.KseiKepemilikanEfek.price * db.KseiKepemilikanEfek.foreign_fd).label('foreign_fd'),
				func.sum(db.KseiKepemilikanEfek.price * db.KseiKepemilikanEfek.foreign_ot).label('foreign_ot'),
				func.sum(db.KseiKepemilikanEfek.price * db.KseiKepemilikanEfek.foreign_total).label('foreign_total'), # type: ignore
			).distinct(db.KseiKepemilikanEfek.date)\
			.group_by(db.KseiKepemilikanEfek.date)\
			.filter(db.KseiKepemilikanEfek.sectype == 'equity')\
			.filter(db.KseiKepemilikanEfek.date >= self.startdate)\
			.filter(db.KseiKepemilikanEfek.date <= self.enddate)
		if self.stockcode != "composite":
			qry = qry.filter(db.KseiKepemilikanEfek.code == self.stockcode)
			
		return pd.read_sql(sql=qry.statement, con=dbs.bind, parse_dates=['date']).reset_index(drop=True).set_index("date").sort_index() # type: ignore
	
	async def __get_data_scripless(self, list_date:list, dbs:db.Session = next(db.get_dbs())) -> pd.DataFrame:
		# Query table stockdata: tradebleshares divided by listedshares for filtercode each list_date
		qry = dbs.query(db.StockData.date, db.StockData.tradebleshares, db.StockData.listedshares).filter(db.StockData.code == self.stockcode).filter(db.StockData.date.in_(list_date)) # type: ignore
		data_scripless = pd.read_sql(sql=qry.statement, con=dbs.bind, parse_dates=['date']).reset_index(drop=True).set_index("date").sort_index() # type: ignore
		data_scripless["scripless_ratio"] = data_scripless["tradebleshares"] / data_scripless["listedshares"]
		return data_scripless[["scripless_ratio"]]

	async def get(self, dbs:db.Session = next(db.get_dbs())) -> HoldingComposition:
		"""
		Process the data
		"""
		# ==========
		# GET DATA
		# ==========
		# Get dataparam
		dataparam = await self.__get_param(dbs=dbs)
		# Replace startdate to the first day of the month of today - month(default_months_range) if startdate is None
		if self.startdate is None:
			self.startdate = datetime.date.today().replace(day=1) - relativedelta(months=int(dataparam["default_months_range"]))

		# Get the data KSEI
		data_ksei = await self.__get_data_ksei(dbs=dbs)

		# Get data scripless ratio for each date from data_ksei and convert to datetime.date
		list_date = data_ksei.index.to_pydatetime().tolist() # type: ignore
		data_scripless = await self.__get_data_scripless(list_date=list_date, dbs=dbs)

		# ==========
		# CALCULATION
		# ==========
		# Group for each row by self.categorization keys, that each keys has array of value that a column name of data
		holding_composition = pd.DataFrame()
		for key in self.categorization.value.keys():
			holding_composition[key] = data_ksei[self.categorization.value[key]].sum(axis=1)

		# Calculate the percentage of total each row
		holding_composition = holding_composition.div(holding_composition.sum(axis=1), axis=0)

		# Append data_scripless[scripless_ratio] column to holding_composition
		holding_composition = holding_composition.join(data_scripless, how="left")
		holding_composition['scripless_ratio'] = holding_composition['scripless_ratio'].fillna(0)

		# Return the result
		self.holding_composition = holding_composition
		return self
