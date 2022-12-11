"""
DB dataparam
	- param: screener_status: 0 (not running), 1 (running)
	- param: screener_last_update: datetime, nan
	- param: screener_period: integer, 5
	- param: screener_minvalue: float, 0
	- param: screener_minfreq: integer, 0
	- param: screener_minwbuy: float, 0
	- param: screener_minwsell: float, 0
DB screener_result
	- date
	- screener_method
	- code
	- close
	- money flow
	- Proportion
	- correlation
	- vwap
"""
from __future__ import annotations
import datetime
import pandas as pd
import numpy as np
import database as db
import dependencies as dp

class ScreenerBase():
	def __init__(self,
		analysis_method: str,
		period: int | None = None,
		start_date: datetime.date | None = None,
		end_date: datetime.date | None = None,
		) -> None:
		
		self.analysis_method: str = analysis_method
		self.period: int | None = period
		self.start_date: datetime.date | None = start_date
		self.end_date: datetime.date | None = end_date

		self.df: None = None

	# get default screener parameters
	async def get_default_param(self,
		analysis_method: dp.AnalysisMethod,
		dbs: db.Session | None = next(db.get_dbs()),
		) -> pd.Series:
		assert dbs is not None

		if analysis_method == dp.AnalysisMethod.foreign:
			method_filter:str = "default_ff_%"
		elif analysis_method == dp.AnalysisMethod.broker:
			method_filter:str = "default_bf_%"
		else:
			raise ValueError(f"analysis_method {analysis_method} is not supported")
		
		qry = dbs.query(db.DataParam, db.DataParam.value)\
			.filter(db.DataParam.param.like("default_screener_%") | \
					db.DataParam.param.like(method_filter))

		return pd.Series(pd.read_sql(sql=qry.statement, con=dbs.bind).set_index("param")['value'])

	# get list_stock that should be analyzed
	async def get_filtered_stock(self,
		analysis_method: dp.AnalysisMethod,
		screener_minvalue: float,
		screener_minfreq: int,
		screener_minwbuy: float,
		screener_minwsell: float,
		dbs: db.Session | None = next(db.get_dbs()),
		) -> pd.DataFrame:
		assert dbs is not None
		
		if analysis_method == dp.AnalysisMethod.foreign:
			qry = dbs.query(db.ListStock)\
				.filter(db.ListStock.value > screener_minvalue, \
						db.ListStock.frequency > screener_minfreq, \
						db.ListStock.foreignsellval > screener_minwbuy, \
						db.ListStock.foreignbuyval > screener_minwsell)

		elif analysis_method == dp.AnalysisMethod.broker:
			qry = dbs.query(db.ListStock)

		else:
			raise ValueError(f"analysis_method {analysis_method} is not supported")

		return pd.read_sql(sql=qry.statement, con=dbs.bind)

class ScreenerMostAccum(ScreenerBase):
	def __init__(self,
		period: int | None = None,
		start_date: datetime.date | None = None,
		end_date: datetime.date | None = None,
		) -> None:
		
		super().__init__(dp.ScreenerList.most_accumulated, period, start_date, end_date)

	async def screen(self,) -> pd.DataFrame:
		return pd.DataFrame()

	

	# get ranked, filtered, and pre-calculated indicator data of selected list_stock
	
	# output: list of stock code and their indicator data