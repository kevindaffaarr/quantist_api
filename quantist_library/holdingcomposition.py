from __future__ import annotations
from enum import Enum

import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd

import database as db
import dependencies as dp

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

		self.stockcode = stockcode.lower()
		self.startdate = startdate
		self.enddate = enddate
		self.categorization = categorization

		# Validation startdate and enddate
		if isinstance(self.startdate, datetime.date) and self.startdate < datetime.date(2015, 1, 1):
			raise ValueError("Startdate must be >= 2015-01-01")
		if self.enddate > datetime.date.today().replace(day=1) - datetime.timedelta(days=1):
			raise ValueError("Enddate must be <= last month")
	
	async def __get_param(self, dbs:db.Session = next(db.get_dbs())) -> pd.Series:
		qry = dbs.query(db.DataParam.param, db.DataParam.value)\
			.filter(db.DataParam.param == 'default_months_range')
		return pd.Series(pd.read_sql(sql=qry.statement, con=dbs.bind).set_index("param")["value"])

	async def __get_data_ksei(self, dbs:db.Session = next(db.get_dbs())) -> pd.DataFrame:
		"""
		Get the data from database filtered by stockcode, sectype, and date range
		"""
		# Get the data from database
		qry = dbs.query(db.KseiKepemilikanEfek)\
			.filter(db.KseiKepemilikanEfek.code == self.stockcode)\
			.filter(db.KseiKepemilikanEfek.sectype == 'equity')\
			.filter(db.KseiKepemilikanEfek.date >= self.startdate)\
			.filter(db.KseiKepemilikanEfek.date <= self.enddate)
		
		return pd.read_sql(sql=qry.statement, con=dbs.bind, parse_dates=['date']).reset_index(drop=True).set_index("date").sort_index()

	async def __get_data_scripless(self, list_date:list, dbs:db.Session = next(db.get_dbs())) -> pd.DataFrame:
		# Query table stockdata: tradebleshares divided by listedshares for self.stockcode each list_date
		qry = dbs.query(db.StockData.date, db.StockData.tradebleshares, db.StockData.listedshares)\
			.filter(db.StockData.code == self.stockcode)\
			.filter(db.StockData.date.in_(list_date))
		data_scripless = pd.read_sql(sql=qry.statement, con=dbs.bind, parse_dates=['date']).reset_index(drop=True).set_index("date").sort_index()
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

		# Get data scripless ratio for each date in data_ksei
		list_date = data_ksei.index.tolist()
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

		# Return the result
		self.holding_composition = holding_composition
		return self
