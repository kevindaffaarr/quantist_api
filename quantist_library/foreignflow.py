from __future__ import annotations
from typing import Literal
import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
from sqlalchemy.sql import func, desc, asc
import database as db
import dependencies as dp
from quantist_library import genchart

class StockFFFull():
	def __init__(self,
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
		dbs: db.Session = next(db.get_dbs())
		) -> None:
		self.stockcode = stockcode
		self.startdate = startdate
		self.enddate = enddate
		self.period_mf = period_mf
		self.period_prop = period_prop
		self.period_pricecorrel = period_pricecorrel
		self.period_mapricecorrel = period_mapricecorrel
		self.period_vwap = period_vwap
		self.pow_high_prop = pow_high_prop
		self.pow_high_pricecorrel = pow_high_pricecorrel
		self.pow_high_mapricecorrel = pow_high_mapricecorrel
		self.pow_medium_prop = pow_medium_prop
		self.pow_medium_pricecorrel = pow_medium_pricecorrel
		self.pow_medium_mapricecorrel = pow_medium_mapricecorrel
		self.dbs = dbs

		self.type = None
		self.wf_indicators = pd.DataFrame()

	async def fit(self) -> StockFFFull:
		# Get defaults value
		default_ff = await self.__get_default_ff(self.dbs)

		# Check Does Stock Code is composite
		self.stockcode = (str(default_ff['default_stockcode']) if self.stockcode is None else self.stockcode).lower() # Default stock is parameterized, may become a branding or endorsement option
		if self.stockcode == 'composite':
			self.type = 'composite'
		else:
			qry = self.dbs.query(db.ListStock.code).filter(db.ListStock.code == self.stockcode)
			row = pd.read_sql(sql=qry.statement, con=self.dbs.bind)
			if len(row) != 0:
				self.type = 'stock'
			else:
				raise KeyError("There is no such stock code in the database.")
		
		# Data Parameter
		default_months_range = int(default_ff['default_months_range']) if self.startdate is None else 0
		self.enddate = datetime.date.today() if self.enddate is None else self.enddate
		self.startdate = self.enddate - relativedelta(months=default_months_range) if self.startdate is None else self.startdate
		self.period_mf = int(default_ff['default_ff_period_mf']) if self.period_mf is None else self.period_mf
		self.period_prop = int(default_ff['default_ff_period_prop']) if self.period_prop is None else self.period_prop
		self.period_pricecorrel = int(default_ff['default_ff_period_pricecorrel']) if self.period_pricecorrel is None else self.period_pricecorrel
		self.period_mapricecorrel = int(default_ff['default_ff_period_mapricecorrel']) if self.period_mapricecorrel is None else self.period_mapricecorrel
		self.period_vwap = int(default_ff['default_ff_period_vwap']) if self.period_vwap is None else self.period_vwap
		self.pow_high_prop = int(default_ff['default_ff_pow_high_prop']) if self.pow_high_prop is None else self.pow_high_prop
		self.pow_high_pricecorrel = int(default_ff['default_ff_pow_high_pricecorrel']) if self.pow_high_pricecorrel is None else self.pow_high_pricecorrel
		self.pow_high_mapricecorrel = int(default_ff['default_ff_pow_high_mapricecorrel']) if self.pow_high_mapricecorrel is None else self.pow_high_mapricecorrel
		self.pow_medium_prop = int(default_ff['default_ff_pow_medium_prop']) if self.pow_medium_prop is None else self.pow_medium_prop
		self.pow_medium_pricecorrel = int(default_ff['default_ff_pow_medium_pricecorrel']) if self.pow_medium_pricecorrel is None else self.pow_medium_pricecorrel
		self.pow_medium_mapricecorrel = int(default_ff['default_ff_pow_medium_mapricecorrel']) if self.pow_medium_mapricecorrel is None else self.pow_medium_mapricecorrel
		preoffset_period_param = max(self.period_mf,self.period_prop,self.period_pricecorrel,(self.period_mapricecorrel+self.period_vwap))-1

		# Raw Data
		if self.type == 'composite':
			raw_data = await self.__get_composite_raw_data(self.dbs,\
				self.startdate,self.enddate,\
				default_months_range,preoffset_period_param)
		else: #self.type == 'stock'
			raw_data = await self.__get_stock_raw_data(self.dbs,self.stockcode,\
				self.startdate,self.enddate,\
				default_months_range,preoffset_period_param)
		
		# Foreign Flow Indicators
		self.wf_indicators = await self.calc_wf_indicators(raw_data,\
			self.period_mf,self.period_prop,self.period_pricecorrel,self.period_mapricecorrel,self.period_vwap,\
			self.pow_high_prop,self.pow_high_pricecorrel,self.pow_high_mapricecorrel,self.pow_medium_prop,\
			self.pow_medium_pricecorrel,self.pow_medium_mapricecorrel,\
			preoffset_period_param
			)

		# Not to be ran inside init, but just as a method that return plotly fig
		# self.chart(media_type="json")

		return self
		
	async def __get_default_ff(self, dbs:db.Session = next(db.get_dbs())) -> pd.Series:

		qry = dbs.query(db.DataParam.param, db.DataParam.value)\
			.filter((db.DataParam.param.like("default_ff_%")) | \
				(db.DataParam.param.like("default_stockcode")) | \
				(db.DataParam.param.like("default_months_range")))
		return pd.Series(pd.read_sql(sql=qry.statement, con=dbs.bind).set_index("param")['value'])

	async def __get_stock_raw_data(self,
		dbs:db.Session = next(db.get_dbs()),
		stockcode: str = ...,
		startdate: datetime.date | None = None,
		enddate: datetime.date = datetime.date.today(),
		default_months_range:int | None = None,
		preoffset_period_param: int = 50
		) -> pd.DataFrame:

		# Define startdate to 1 year before enddate if startdate is None
		if startdate is None:
			startdate = enddate - relativedelta(months=default_months_range)
		# Query Definition
		qry = dbs.query(
			db.StockData.date,
			db.StockData.previous,
			db.StockData.openprice,
			db.StockData.high,
			db.StockData.low,
			db.StockData.close,
			db.StockData.value,
			db.StockData.foreignbuy,
			db.StockData.foreignsell
		).filter(db.StockData.code == stockcode)
		
		# Main Query
		qry_main = qry.filter(db.StockData.date.between(startdate, enddate)).order_by(db.StockData.date.asc())
		
		# Main Query Fetching
		raw_data_main = pd.read_sql(sql=qry_main.statement, con=dbs.bind, parse_dates=["date"])\
			.reset_index(drop=True).set_index("date")
		
		# Check how many row is returned
		if raw_data_main.shape[0] == 0:
			raise ValueError("No data available inside date range")
		
		# Update self.startdate and self.enddate to available date in database
		self.startdate = raw_data_main.index[0].date() # type: ignore # Assign self.startdate to new defined startdate
		self.enddate = raw_data_main.index[-1].date() # type: ignore # Assign self.enddate to new defined enddate
		
		# Pre-Data Query
		startdate = self.startdate
		qry_pre = qry.filter(db.StockData.date < startdate)\
			.order_by(db.StockData.date.desc())\
			.limit(preoffset_period_param)\
			.subquery()
		qry_pre = dbs.query(qry_pre).order_by(qry_pre.c.date.asc())

		# Pre-Data Query Fetching
		raw_data_pre = pd.read_sql(sql=qry_pre.statement, con=dbs.bind, parse_dates=["date"])\
			.reset_index(drop=True).set_index('date')

		# Concatenate Pre and Main Query
		raw_data = pd.concat([raw_data_pre,raw_data_main])

		# Data Cleansing: zero openprice replace with previous
		raw_data['openprice'] = raw_data['openprice'].mask(raw_data['openprice'].eq(0),raw_data['previous'])

		# End of Method: Return or Assign Attribute
		return raw_data

	async def __get_composite_raw_data(self, 
		dbs:db.Session = next(db.get_dbs()),
		startdate: datetime.date | None = None,
		enddate: datetime.date = datetime.date.today(),
		default_months_range:int | None = None,
		preoffset_period_param: int = 50
		) -> pd.DataFrame:

		# Define startdate to 1 year before enddate if startdate is None
		if startdate is None:
			startdate = enddate - relativedelta(months=default_months_range)
		# Query Definition
		qry = dbs.query(
			db.IndexData.date,
			db.IndexData.previous.label('openprice'),
			db.IndexData.highest.label('high'),
			db.IndexData.lowest.label('low'),
			db.IndexData.close,
			db.IndexData.value,
			db.IndexTransactionCompositeForeign.foreignbuyval.label('foreignbuy'),
			db.IndexTransactionCompositeForeign.foreignsellval.label('foreignsell')
		).filter(db.IndexData.code == 'composite')\
		.join(db.IndexTransactionCompositeForeign,
			db.IndexTransactionCompositeForeign.date == db.IndexData.date
		)
		
		# Main Query
		qry_main = qry.filter(db.IndexData.date.between(startdate, enddate)).order_by(db.IndexData.date.asc())
		
		# Main Query Fetching
		raw_data_main = pd.read_sql(sql=qry_main.statement, con=dbs.bind, parse_dates=["date"])\
			.reset_index(drop=True).set_index('date')
		
		# Check how many row is returned
		if raw_data_main.shape[0] == 0:
			raise ValueError("No data available inside date range")
		
		# Update self.startdate and self.enddate to available date in database
		self.startdate = raw_data_main.index[0].date() # type: ignore # Assign self.startdate to new defined startdate
		self.enddate = raw_data_main.index[-1].date() # type: ignore # Assign self.enddate to new defined enddate
		
		# Pre-Data Query
		startdate = self.startdate
		qry_pre = qry.filter(db.IndexData.date < startdate)\
			.order_by(db.IndexData.date.desc())\
			.limit(preoffset_period_param).subquery()
		qry_pre = dbs.query(qry_pre).order_by(qry_pre.c.date.asc())

		# Pre-Data Query Fetching
		raw_data_pre = pd.read_sql(sql=qry_pre.statement, con=dbs.bind, parse_dates=["date"])\
			.reset_index(drop=True).set_index('date')

		# Concatenate Pre and Main Query
		raw_data = pd.concat([raw_data_pre,raw_data_main])

		# Convert Foreign Transaction Value to Volume
		raw_data["foreignbuy"] = raw_data["foreignbuy"]/raw_data["close"]
		raw_data["foreignsell"] = raw_data["foreignsell"]/raw_data["close"]

		# End of Method: Return or Assign Attribute
		return raw_data

	async def calc_wf_indicators(self,
		raw_data: pd.DataFrame,
		period_mf: int = 1,
		period_prop: int = 10,
		period_pricecorrel: int = 10,
		period_mapricecorrel: int = 100,
		period_vwap:int = 21,
		pow_high_prop: int = 40,
		pow_high_pricecorrel: int = 50,
		pow_high_mapricecorrel: int = 30,
		pow_medium_prop: int = 20,
		pow_medium_pricecorrel: int = 30,
		pow_medium_mapricecorrel: int = 30,
		preoffset_period_param: int = 0
		) -> pd.DataFrame:
		# Define fbval, fsval, netvol, netval
		raw_data['fbval'] = raw_data['close']*raw_data['foreignbuy']
		raw_data['fsval'] = raw_data['close']*raw_data['foreignsell']
		raw_data['netvol'] = raw_data['foreignbuy']-raw_data['foreignsell']
		raw_data['netval'] = raw_data['fbval']-raw_data['fsval']
		
		# FF
		raw_data['volflow'] = raw_data['netvol'].cumsum()

		# MF
		raw_data['mf'] = raw_data['netval'].rolling(window=period_mf).sum()

		# Prop
		raw_data['prop'] = (raw_data['fbval']+raw_data['fsval']).rolling(window=period_prop).sum()\
			/(raw_data['value'].rolling(window=period_prop).sum()*2)

		# FNetProp
		raw_data['netprop'] = abs(raw_data['netval']).rolling(window=period_prop).sum()\
			/(raw_data['value'].rolling(window=period_prop).sum()*2)

		# pricecorrel
		raw_data['pricecorrel'] = raw_data['close'].rolling(window=period_pricecorrel)\
			.corr(raw_data['volflow'])
		
		# MAPriceCorrel
		raw_data['mapricecorrel'] = raw_data['pricecorrel'].rolling(window=period_mapricecorrel).mean()
		
		# VWAP
		raw_data['vwap'] = (raw_data['netval'].rolling(window=period_vwap).apply(lambda x: x[x>0].sum()))\
			/(raw_data['netvol'].rolling(window=period_vwap).apply(lambda x: x[x>0].sum()))
		
		raw_data['vwap'] = raw_data['vwap'].mask(raw_data['vwap'].le(0)).ffill()
		
		# Pow
		raw_data['pow'] = \
			np.where(
				(raw_data["prop"]>(pow_high_prop/100)) & \
				(raw_data['pricecorrel']>(pow_high_pricecorrel/100)) & \
				(raw_data['mapricecorrel']>(pow_high_mapricecorrel/100)), \
				3,
				np.where(
					(raw_data["prop"]>(pow_medium_prop/100)) & \
					(raw_data['pricecorrel']>(pow_medium_pricecorrel/100)) & \
					(raw_data['mapricecorrel']>(pow_medium_mapricecorrel/100)), \
					2, \
					1
				)
			)

		# End of Method: Return Processed Raw Data to FF Indicators
		return raw_data.loc[self.startdate:self.enddate]
	
	async def chart(self,media_type:str | None = None):
		assert self.stockcode is not None
		fig = await genchart.quantist_stock_chart(
			stockcode=self.stockcode,
			wf_indicators=self.wf_indicators,
			analysis_method=dp.AnalysisMethod.foreign,
			period_prop=self.period_prop,
			period_pricecorrel=self.period_pricecorrel,
			period_mapricecorrel=self.period_mapricecorrel,
			period_vwap=self.period_vwap,
			)
		if media_type in ["png","jpeg","jpg","webp","svg"]:
			return await genchart.fig_to_image(fig,media_type)
		elif media_type == "json":
			return await genchart.fig_to_json(fig)
		else:
			return fig
	
class ForeignRadar():
	def __init__(self,
		startdate: datetime.date | None = None,
		enddate: datetime.date = datetime.date.today(),
		y_axis_type: dp.ListRadarType = dp.ListRadarType.correlation,
		stockcode_excludes: set[str] = set(),
		include_composite: bool = False,
		screener_min_value: int | None = None,
		screener_min_frequency: int | None = None,
		screener_min_prop:int | None = None,
		period_mf: int | None = None,
		period_pricecorrel: int | None = None,
		dbs: db.Session = next(db.get_dbs())
		) -> None:
		self.startdate = startdate
		self.enddate = enddate
		self.y_axis_type = y_axis_type
		self.stockcode_excludes = stockcode_excludes
		self.include_composite = include_composite
		self.screener_min_value = screener_min_value
		self.screener_min_frequency = screener_min_frequency
		self.screener_min_prop = screener_min_prop
		self.period_mf = period_mf
		self.period_pricecorrel = period_pricecorrel
		self.dbs = dbs

		self.radar_indicators =  pd.DataFrame()

	async def fit(self) -> ForeignRadar:
		# Get default value of parameter
		default_radar = await self._get_default_radar()

		self.period_mf = int(default_radar['default_radar_period_mf']) if self.startdate is None else None
		self.period_pricecorrel = int(default_radar['default_radar_period_pricecorrel']) if self.startdate is None else None
		self.screener_min_value = int(default_radar['default_screener_min_value']) if self.screener_min_value is None else self.screener_min_value
		self.screener_min_frequency = int(default_radar['default_screener_min_frequency']) if self.screener_min_frequency is None else self.screener_min_frequency
		self.screener_min_prop = int(default_radar['default_screener_min_prop']) if self.screener_min_prop is None else self.screener_min_prop
		
		# Get filtered stockcodes
		filtered_stockcodes = await self._get_stockcodes(
			screener_min_value=self.screener_min_value,
			screener_min_frequency=self.screener_min_frequency,
			screener_min_prop=self.screener_min_prop,
			stockcode_excludes=self.stockcode_excludes,
			dbs=self.dbs)

		# Get raw data
		if self.startdate is None:
			assert self.period_mf is not None
			assert self.period_pricecorrel is not None
			bar_range = max(self.period_mf,self.period_pricecorrel)
		else:
			bar_range = None
		stocks_raw_data = await self.__get_stocks_raw_data(
			startdate=self.startdate,
			enddate=self.enddate,
			filtered_stockcodes=filtered_stockcodes,
			bar_range=bar_range,
			dbs=self.dbs)

		# Check how many row is returned
		if stocks_raw_data.shape[0] == 0:
			raise ValueError("No data available inside date range")
		
		# Set startdate and enddate based on data availability
		self.startdate = stocks_raw_data['date'].min().date()
		self.enddate = stocks_raw_data['date'].max().date()

		# Calc Radar Indicators: last pricecorrel OR last changepercentage
		self.radar_indicators = await self.calc_radar_indicators(\
			stocks_raw_data=stocks_raw_data,y_axis_type=self.y_axis_type)

		if self.include_composite is True:
			composite_raw_data = await self.__get_composite_raw_data(
				startdate=self.startdate,
				enddate=self.enddate,
				bar_range=bar_range,
				dbs=self.dbs
			)
			composite_radar_indicators = await self.calc_radar_indicators(\
				stocks_raw_data=composite_raw_data,y_axis_type=self.y_axis_type)
			
			self.radar_indicators = pd.concat([self.radar_indicators,composite_radar_indicators])

		# Not to be ran inside init, but just as a method that return plotly fig
		# self.chart(media_type="json")

		return self
		
	async def _get_default_radar(self, dbs:db.Session = next(db.get_dbs())) -> pd.Series:
		qry = dbs.query(db.DataParam.param, db.DataParam.value)\
			.filter((db.DataParam.param.like("default_radar_%")) | (db.DataParam.param.like("default_screener_%")) | (db.DataParam.param.like("default_ff_%")))
		
		return pd.Series(pd.read_sql(sql=qry.statement, con=dbs.bind).set_index("param")['value'])
		
	
	async def _get_stockcodes(self,
		screener_min_value: int = 5000000000,
		screener_min_frequency: int = 1000,
		screener_min_prop:int = 0,
		stockcode_excludes: set[str] = set(),
		dbs: db.Session = next(db.get_dbs())
		) -> pd.Series:
		"""
		Get filtered stockcodes
		Filtered by:value>screener_min_value, frequency>screener_min_frequency 
					foreignbuyval>0, foreignsellval>0,
					stockcode_excludes
		"""
		# Query Definition
		stockcode_excludes_lower = set(x.lower() for x in stockcode_excludes) if stockcode_excludes is not None else set()
		qry = dbs.query(db.ListStock.code)\
			.filter((db.ListStock.value > screener_min_value) &
					(db.ListStock.frequency > screener_min_frequency) &
					(db.ListStock.foreignbuyval > 0) &
					(db.ListStock.foreignsellval > 0) &
					(((db.ListStock.foreignsellval+db.ListStock.foreignbuyval)/(db.ListStock.value*2)) > (screener_min_prop/100)) &
					(db.ListStock.code.not_in(stockcode_excludes_lower)))
		
		# Query Fetching: filtered_stockcodes
		return pd.Series(pd.read_sql(sql=qry.statement,con=dbs.bind).reset_index(drop=True)['code'])

	async def __get_stocks_raw_data(self,
		startdate: datetime.date | None = None,
		enddate: datetime.date = datetime.date.today(),
		filtered_stockcodes:pd.Series = ...,
		bar_range:int | None = 5,
		dbs: db.Session = next(db.get_dbs())
		) -> pd.DataFrame:

		# Jika belum ada startdate, maka perlu ditentukan batas mulainya
		if startdate is None:
			# Ambil tanggal-tanggal pada BBCA yang meliputi enddate dan limit hingga bar_range
			sub_qry = dbs.query(db.StockData.date)\
				.filter(db.StockData.code == 'bbca')\
				.filter(db.StockData.date <= enddate)\
				.order_by(db.StockData.date.desc())\
				.limit(bar_range)\
				.subquery()
			# Ambil tanggal minimum yang pada akhirnya akan dijadikan sebagai startdate
			sub_qry = dbs.query(func.min(sub_qry.c.date)).scalar_subquery()
			
			startdate = sub_qry
			
		# else:
			# startdate = startdate

		# Main Query
		qry = dbs.query(db.StockData.code,
			db.StockData.date,
			db.StockData.close,
			db.StockData.foreignbuy,
			db.StockData.foreignsell
			)\
			.filter(db.StockData.code.in_(filtered_stockcodes.to_list()))\
			.filter(db.StockData.date.between(startdate,enddate))\
			.order_by(db.StockData.code.asc(),db.StockData.date.asc())
			
		# Query Fetching: stocks raw data
		return pd.read_sql(sql=qry.statement,con=dbs.bind,parse_dates=["date"])\
			.reset_index(drop=True)
	
	async def __get_composite_raw_data(self,
		startdate:datetime.date | None =None,
		enddate:datetime.date | None =None,
		bar_range:int | None = 5,
		dbs:db.Session = next(db.get_dbs())
		) ->  pd.DataFrame:

		# Jika belum ada startdate, maka perlu ditentukan batas mulainya
		if startdate is None:
			# Ambil tanggal-tanggal pada composite yang meliputi enddate dan limit hingga bar_range
			sub_qry = dbs.query(db.IndexData.date)\
				.filter(db.IndexData.code == 'composite')\
				.filter(db.IndexData.date <= enddate)\
				.order_by(db.IndexData.date.desc())\
				.limit(bar_range)\
				.subquery()
			# Ambil tanggal minimum yang pada akhirnya akan dijadikan sebagai startdate
			sub_qry = dbs.query(func.min(sub_qry.c.date)).scalar_subquery()
			
			startdate = sub_qry

		# else:
			# startdate = startdate
		
		# Main Query
		qry = dbs.query(
			db.IndexData.code,
			db.IndexData.date,
			db.IndexData.close,
			(db.IndexTransactionCompositeForeign.foreignbuyval/db.IndexData.close).label('foreignbuy'), # type: ignore
			(db.IndexTransactionCompositeForeign.foreignsellval/db.IndexData.close).label('foreignsell') # type: ignore
			).\
			filter(db.IndexData.code=="composite")\
			.filter(db.IndexData.date.between(startdate,enddate))\
			.join(db.IndexTransactionCompositeForeign,
				db.IndexTransactionCompositeForeign.date == db.IndexData.date)\
			.order_by(db.IndexData.date.asc())

		# Query Fetching: composite raw data
		composite_raw_data = pd.read_sql(sql=qry.statement,con=dbs.bind,parse_dates=["date"])\
			.reset_index(drop=True)
		return composite_raw_data

	async def calc_radar_indicators(self,
		stocks_raw_data:pd.DataFrame,
		y_axis_type:dp.ListRadarType = dp.ListRadarType.correlation,
		) -> pd.DataFrame:
		
		radar_indicators = pd.DataFrame()

		# Y axis: mf
		stocks_raw_data['netval'] = stocks_raw_data['close']*\
			(stocks_raw_data['foreignbuy']-stocks_raw_data['foreignsell'])
		radar_indicators['mf'] = stocks_raw_data.groupby(by='code')['netval'].sum()

		# X axis:
		if y_axis_type == dp.ListRadarType.correlation:
			# NetVol
			stocks_raw_data['netvol'] = stocks_raw_data['foreignbuy']-stocks_raw_data['foreignsell']
			# FF
			stocks_raw_data['fvolflow'] = stocks_raw_data.groupby('code')['netvol'].cumsum()
			# pricecorrel
			radar_indicators[y_axis_type] = stocks_raw_data.groupby('code')[['fvolflow','close']].corr(method='pearson').iloc[0::2,-1].droplevel(1)
			# radar_indicators[y_axis_type] = stocks_raw_data.groupby(by='code')['fvolflow']\
			# 	.corr(stocks_raw_data['close'])
				
		elif y_axis_type == dp.ListRadarType.changepercentage:
			radar_indicators[y_axis_type] = \
				(stocks_raw_data.groupby(by='code')['close'].nth([-1])- \
				stocks_raw_data.groupby(by='code')['close'].nth([0]))/ \
				stocks_raw_data.groupby(by='code')['close'].nth([0])
		else:
			raise ValueError("Not a valid radar type")
		
		return radar_indicators
	
	async def chart(self,media_type:str | None = None):
		assert self.startdate is not None

		fig = await genchart.radar_chart(
			startdate=self.startdate,enddate=self.enddate,
			y_axis_type=self.y_axis_type,
			method="Foreign",
			radar_indicators=self.radar_indicators
		)
		if media_type in ["png","jpeg","jpg","webp","svg"]:
			return await genchart.fig_to_image(fig,media_type)
		elif media_type == "json":
			return await genchart.fig_to_json(fig)
		else:
			return fig

class ScreenerBase(ForeignRadar):
	def __init__(self,
		startdate: datetime.date | None = None,
		enddate: datetime.date = datetime.date.today(),
		screener_period: int | None = None,
		stockcode_excludes: set[str] = set(),
		screener_min_value: int | None = None,
		screener_min_frequency: int | None = None,
		screener_min_prop:int | None = None,
		period_mf: int | None = None,
		period_pricecorrel: int | None = None,
		dbs: db.Session = next(db.get_dbs())		
		) -> None:

		super().__init__(
			startdate = startdate,
			enddate = enddate,
			stockcode_excludes = stockcode_excludes,
			screener_min_value = screener_min_value,
			screener_min_frequency = screener_min_frequency,
			screener_min_prop = screener_min_prop,
			period_mf = period_mf,
			period_pricecorrel = period_pricecorrel,
			dbs = dbs,
		)

		self.screener_period = screener_period

	async def _fit_base(self) -> ScreenerBase:
		# get default param radar
		self.default_radar = await super()._get_default_radar()
		self.period_mf = int(self.default_radar['default_radar_period_mf'])
		self.period_pricecorrel = int(self.default_radar['default_radar_period_pricecorrel'])
		self.screener_min_value = int(self.default_radar['default_screener_min_value']) if self.screener_min_value is None else self.screener_min_value
		self.screener_min_frequency = int(self.default_radar['default_screener_min_frequency']) if self.screener_min_frequency is None else self.screener_min_frequency
		self.screener_min_prop = int(self.default_radar['default_screener_min_prop']) if self.screener_min_prop is None else self.screener_min_prop

		# Define startdate
		if self.startdate is None:
			assert self.period_mf is not None
			assert self.period_pricecorrel is not None
			if self.screener_period:
				self.bar_range = self.screener_period
			else:
				self.bar_range = max(self.period_mf,self.period_pricecorrel)

			# get date from BBCA from enddate to limit as much as bar range so we got the startdate
			sub_qry = self.dbs.query(db.StockData.date
				).filter(db.StockData.code == 'bbca'
				).filter(db.StockData.date <= self.enddate
				).order_by(db.StockData.date.desc()
				).limit(self.bar_range
				).subquery()
			sub_qry = self.dbs.query(func.min(sub_qry.c.date)).scalar_subquery()

			self.startdate = sub_qry
		else:
			self.bar_range = None
			# self.startdate = self.startdate

		# get list_stock that should be analyzed
		self.filtered_stockcodes = await super()._get_stockcodes(
			screener_min_value=self.screener_min_value,
			screener_min_frequency=self.screener_min_frequency,
			screener_min_prop=self.screener_min_prop,
			stockcode_excludes=self.stockcode_excludes,
			dbs=self.dbs
		)

		return self

class ScreenerMoneyFlow(ScreenerBase):
	"""
	Get the most accumulated or distributed stock based on money flow
	between startdate and enddate or within screener_period
	output: n_stockcodes of stockcodes
	"""
	def __init__(self,
		accum_or_distri: Literal[dp.ScreenerList.most_accumulated,dp.ScreenerList.most_distributed] = dp.ScreenerList.most_accumulated,
		n_stockcodes: int = 10,
		startdate: datetime.date | None = None,
		enddate: datetime.date = datetime.date.today(),
		screener_period: int | None = None,
		stockcode_excludes: set[str] = set(),
		screener_min_value: int | None = None,
		screener_min_frequency: int | None = None,
		screener_min_prop:int | None = None,
		dbs: db.Session = next(db.get_dbs())
		) -> None:
		assert accum_or_distri in [dp.ScreenerList.most_accumulated, dp.ScreenerList.most_distributed], f'accum_or_distri must be {dp.ScreenerList.most_accumulated.value} or {dp.ScreenerList.most_distributed.value}'

		super().__init__(
			startdate = startdate,
			enddate = enddate,
			stockcode_excludes = stockcode_excludes,
			screener_min_value = screener_min_value,
			screener_min_frequency = screener_min_frequency,
			screener_min_prop = screener_min_prop,
			dbs = dbs,
		)

		self.accum_or_distri = accum_or_distri
		self.n_stockcodes = n_stockcodes
		self.screener_period = screener_period
	
	async def screen(self) -> ScreenerMoneyFlow:
		# Get default param: self.period_mf, self.period_pricecorrel, self.screener_min_value, self.screener_min_frequency, self.screener_min_prop, self.bar_range
		# Get processed self.startdate, self.filtered_stockcodes
		await super()._fit_base()
		
		# get ranked, filtered, and pre-calculated indicator data of filtered_stockcodes
		self.top_stockcodes = await self._get_mf_top_stockcodes(
			accum_or_distri = self.accum_or_distri,
			n_stockcodes = self.n_stockcodes,
			startdate = self.startdate,
			enddate = self.enddate,
			screener_period=self.screener_period,
			filtered_stockcodes = self.filtered_stockcodes,
			stockcode_excludes = self.stockcode_excludes,
			dbs = self.dbs,
		)

		return self
	
	async def _get_mf_top_stockcodes(self,
		accum_or_distri: dp.ScreenerList = dp.ScreenerList.most_accumulated,
		n_stockcodes: int = 10,
		startdate: datetime.date | None = None,
		enddate: datetime.date = datetime.date.today(),
		screener_period: int | None = None,
		filtered_stockcodes: pd.Series = ...,
		stockcode_excludes: set[str] = ...,
		dbs: db.Session = next(db.get_dbs())
		) -> pd.DataFrame:
		
		# Get the top code
		sub_qry_1 = dbs.query(
			db.StockData.code,
			(func.sum(db.StockData.close*db.StockData.foreignbuy)-func.sum(db.StockData.close*db.StockData.foreignsell)).label('mf')
		).filter(db.StockData.code.in_(filtered_stockcodes)
		).filter(db.StockData.code.notin_(stockcode_excludes)
		).filter(db.StockData.date.between(startdate,enddate)
		).group_by(db.StockData.code)
		if accum_or_distri == dp.ScreenerList.most_distributed:
			sub_qry_1 = sub_qry_1.order_by(asc('mf')).limit(n_stockcodes).subquery()
		else:
			sub_qry_1 = sub_qry_1.order_by(desc('mf')).limit(n_stockcodes).subquery()

		# Get the raw data (just for calculate the pricecorrel)
		qry = dbs.query(
			db.StockData.code,
			db.StockData.date,
			db.StockData.close,
			(db.StockData.foreignbuy-db.StockData.foreignsell).label('netvol'), # type: ignore
			(db.StockData.foreignbuy+db.StockData.foreignsell).label('sumvol'), # type: ignore
			db.StockData.value,
		).join(sub_qry_1, sub_qry_1.c.code == db.StockData.code)

		if (startdate == enddate) or (screener_period == 1):
			# Sub Query for pricecorrel with date from enddate to n rows based on self.period_pricecorrel
			# get date from BBCA from enddate to limit as much as bar range so we got the startdate
			startdate_pricecorrel = self.dbs.query(db.StockData.date
				).filter(db.StockData.code == 'bbca'
				).filter(db.StockData.date <= self.enddate
				).order_by(db.StockData.date.desc()
				).limit(self.period_pricecorrel
				).subquery()
			startdate_pricecorrel = dbs.query(func.min(startdate_pricecorrel.c.date)).scalar_subquery()
			qry = qry.filter(db.StockData.date.between(startdate_pricecorrel,enddate))
		else:
			qry = qry.filter(db.StockData.date.between(startdate,enddate))

		# Query fetching using pandas
		raw_data = pd.read_sql(sql=qry.statement, con=dbs.bind).reset_index(drop=True)\
			.sort_values(['code','date']).set_index(['code','date'])

		# Check raw_data row, if zero, raise error: data not available
		if raw_data.shape[0] == 0:
			raise ValueError(f'No data available between {startdate} and {enddate}')
		
		top_stockcodes = pd.DataFrame()
		top_stockcodes['pricecorrel'] = raw_data.groupby("code")[["close","netvol"]].corr(method='pearson').iloc[0::2,-1].droplevel(1)
		
		# Only get raw_data between startdate and enddate
		if not isinstance(startdate, datetime.date):
			# If screener_period is not None, get n last rows of raw_data for each code group based on screener_period
			if screener_period is not None:
				raw_data = raw_data.groupby("code").tail(screener_period)
			else:
				# Update startdate if startdate not datetime.date to the first date of the raw_data
				startdate = raw_data.index.get_level_values('date').min()
				raw_data = raw_data.loc[(slice(None), slice(startdate, enddate)), :]

		# Calculate MF, Prop, and pricecorrel
		top_stockcodes['mf'] = (raw_data['close']*(raw_data['netvol'])).groupby("code").sum()
		top_stockcodes['prop'] = (raw_data['close']*(raw_data['sumvol'])).groupby("code").sum()/(raw_data['value']*2).groupby("code").sum()
		# replace nan with none
		top_stockcodes = top_stockcodes.replace({np.nan: None})

		# Order by MF
		if accum_or_distri == dp.ScreenerList.most_distributed:
			top_stockcodes = top_stockcodes.sort_values(by='mf', ascending=True)
		else:
			top_stockcodes = top_stockcodes.sort_values(by='mf', ascending=False)
		
		# Update startdate and enddate based on raw_data
		self.startdate = raw_data.index.get_level_values('date').min()
		self.enddate = raw_data.index.get_level_values('date').max()

		return top_stockcodes

class ScreenerVWAP(ScreenerBase):
	"""
	Get the top n stockcodes based on VWAP indicator:
		- Rally (always close > vwap within n days)
		- Around VWAP (close around x% of vwap)
		- Breakout (t_x: close < vwap, t_y: close > vwap, within n days, and now close > vwap)
		- Breakdown (t_x: close > vwap, t_y: close < vwap, within n days, and now close < vwap)
	
	General Rules:
		- Value, Freq, Prop more than min_value, min_frequency, min_prop
	"""
	def __init__(self,
		screener_vwap_criteria: Literal[
			dp.ScreenerList.vwap_rally,
			dp.ScreenerList.vwap_around,
			dp.ScreenerList.vwap_breakout,
			dp.ScreenerList.vwap_breakdown
			] = dp.ScreenerList.vwap_rally,
		n_stockcodes: int = 10,
		startdate: datetime.date | None = None,
		enddate: datetime.date = datetime.date.today(),
		screener_period: int | None = None,
		stockcode_excludes: set[str] = set(),
		percentage_range: float | None = 0.05,
		screener_min_value: int | None = None,
		screener_min_frequency: int | None = None,
		screener_min_prop:int | None = None,
		period_vwap: int | None = None,
		dbs: db.Session = next(db.get_dbs())
		) -> None:
		screener_vwap_criteria_list = [
			dp.ScreenerList.vwap_rally,
			dp.ScreenerList.vwap_around,
			dp.ScreenerList.vwap_breakout,
			dp.ScreenerList.vwap_breakdown
		]
		assert screener_vwap_criteria in screener_vwap_criteria_list, f'screener_vwap_criteria_list must be {screener_vwap_criteria_list}'

		super().__init__(
			startdate = startdate,
			enddate = enddate,
			stockcode_excludes = stockcode_excludes,
			screener_min_value = screener_min_value,
			screener_min_frequency = screener_min_frequency,
			screener_min_prop = screener_min_prop,
			dbs = dbs,
		)

		self.screener_vwap_criteria = screener_vwap_criteria
		self.n_stockcodes = n_stockcodes
		self.screener_period = screener_period
		self.percentage_range = percentage_range
		self.period_vwap = period_vwap

	async def _vwap_prep(self) -> ScreenerVWAP:
		# Get default period_vwap, percentage_range
		self.period_vwap = int(self.default_radar['default_ff_period_vwap']) if self.period_vwap is None else self.period_vwap
		self.percentage_range = float(self.default_radar['default_radar_percentage_range']) if self.percentage_range is None else self.percentage_range
		
		assert isinstance(self.period_vwap, int), f'period_vwap must be integer'

		# Get main data from startdate to enddate
		qry = self.dbs.query(
			db.StockData.date,
			db.StockData.code,
			db.StockData.close,
			db.StockData.foreignbuy,
			db.StockData.foreignsell,
		).filter(db.StockData.code.in_(self.filtered_stockcodes))\
		.filter(db.StockData.code.notin_(self.stockcode_excludes))\
		.filter(db.StockData.date.between(self.startdate,self.enddate))\
		.order_by(db.StockData.code, db.StockData.date)
		
		# Main Query Fetching
		self.raw_data = pd.read_sql(sql=qry.statement, con=self.dbs.bind, parse_dates=['date']).reset_index(drop=True)\
			.sort_values(['code','date']).set_index(['code','date'])
		# Update startdate and enddate based on self.raw_data
		self.startdate = self.raw_data.index.get_level_values('date').min().date()
		self.enddate = self.raw_data.index.get_level_values('date').max().date()

		# Get pre-data startdate
		qry = self.dbs.query(
			db.StockData.date,
		).filter(db.StockData.code == 'bbca')\
		.filter(db.StockData.date < self.startdate)\
		.order_by(db.StockData.date.desc())\
		.limit(self.period_vwap)
		fetch = pd.read_sql(sql=qry.statement, con=self.dbs.bind, parse_dates=['date']).reset_index(drop=True)
		pre_startdate = fetch.iloc[:,0].min()
		# Get pre-data for VWAP
		qry = self.dbs.query(
			db.StockData.date,
			db.StockData.code,
			db.StockData.close,
			db.StockData.foreignbuy,
			db.StockData.foreignsell,
		).filter(db.StockData.code.in_(self.filtered_stockcodes))\
		.filter(db.StockData.code.notin_(self.stockcode_excludes))\
		.filter(db.StockData.date < self.startdate)\
		.filter(db.StockData.date >= pre_startdate)\
		.order_by(db.StockData.code, db.StockData.date)
		pre_raw_data = pd.read_sql(sql=qry.statement, con=self.dbs.bind, parse_dates=['date']).reset_index(drop=True)\
			.sort_values(['code','date']).set_index(['code','date'])
		
		# Concat data
		self.raw_data = pd.concat([pre_raw_data, self.raw_data], axis=0)

		# Calculate VWAP
		self.raw_data['fbval'] = self.raw_data['close']*self.raw_data['foreignbuy']
		self.raw_data['fsval'] = self.raw_data['close']*self.raw_data['foreignsell']
		self.raw_data['netvol'] = self.raw_data['foreignbuy']-self.raw_data['foreignsell']
		self.raw_data['netval'] = self.raw_data['fbval']-self.raw_data['fsval']
		self.raw_data['vwap'] = ((self.raw_data['netval'].groupby(level='code').rolling(window=self.period_vwap).apply(lambda x: x[x>0].sum()))\
			/(self.raw_data['netvol'].groupby(level='code').rolling(window=self.period_vwap).apply(lambda x: x[x>0].sum()))).droplevel(0)

		# Filter only startdate to enddate
		self.raw_data = self.raw_data.loc[
			(self.raw_data.index.get_level_values('date') >= pd.Timestamp(self.startdate)) & \
			(self.raw_data.index.get_level_values('date') <= pd.Timestamp(self.enddate))]

		return self

	async def screen(self) -> ScreenerVWAP:
		# Get default param: self.period_mf, self.period_pricecorrel, self.screener_min_value, self.screener_min_frequency, self.screener_min_prop, self.bar_range
		# Get processed self.startdate, self.filtered_stockcodes
		await super()._fit_base()

		# Get self.raw_data, and update self.startdate, self.enddate
		await self._vwap_prep()

		# Go to get top codes for each screener_vwap_criteria
		if self.screener_vwap_criteria == dp.ScreenerList.vwap_rally:
			stocklist = await self._get_vwap_rally()
		elif self.screener_vwap_criteria == dp.ScreenerList.vwap_around:
			stocklist = await self._get_vwap_around()
		elif self.screener_vwap_criteria == dp.ScreenerList.vwap_breakout:
			stocklist = await self._get_vwap_breakout()
		elif self.screener_vwap_criteria == dp.ScreenerList.vwap_breakdown:
			stocklist = await self._get_vwap_breakdown()
		else:
			raise ValueError(f'Invalid screener_vwap_criteria: {self.screener_vwap_criteria}')
		
		# Get data from stocklist
		await self._get_data_from_stocklist(stocklist)

		# Compile data for top_stockcodes from stocklist and top_data
		self.top_stockcodes = self.top_data[['close','vwap']].groupby(level='code').last()
		self.top_stockcodes['mf'] = self.top_data['netval'].groupby(level='code').sum()
		self.top_stockcodes = self.top_stockcodes.sort_values('mf', ascending=False)
		
		return self

	async def _get_data_from_stocklist(self, stocklist: list) -> ScreenerVWAP:
		# Get data from stocklist
		top_data = self.raw_data.loc[self.raw_data.index.get_level_values('code').isin(stocklist)]
		# Sum netval for each code and get top n_stockcodes
		stocklist = top_data['netval'].groupby(level='code').sum().nlargest(self.n_stockcodes).index.tolist()

		# Get data from stocklist
		top_data = self.raw_data.loc[self.raw_data.index.get_level_values('code').isin(stocklist)]
		
		self.stocklist = stocklist
		self.top_data = top_data

		return self

	async def _get_vwap_rally(self) -> list:
		"""Rally (always close > vwap within n days)"""
		# Get stockcodes with self.raw_data['close'] always self.raw_data['vwap']
		stocklist = (self.raw_data['close'] >= self.raw_data['vwap']).groupby(level='code').all()
		stocklist = stocklist[stocklist].index.tolist()

		return stocklist

	async def _get_vwap_around(self) -> list:
		"""Around VWAP (close around x% of vwap)"""
		assert self.percentage_range is not None
		# Get stockcodes with last self.raw_data['close'] around last self.raw_data['vwap'], within percentage_range
		last_data = self.raw_data[['close','vwap']].groupby(level='code').last()
		stocklist = last_data[(last_data['close'] >= last_data['vwap']*(1-self.percentage_range)) & (last_data['close'] <= last_data['vwap']*(1+self.percentage_range))].index.tolist()

		return stocklist

	async def _get_vwap_breakout(self) -> list:
		"""Breakout (t_(x-1): close < vwap, t_(x): close > vwap, within n days, and now close > vwap)"""
		# Get stockcodes with now close > vwap
		last_data = self.raw_data[['close','vwap']].groupby(level='code').last()
		stocklist = last_data[last_data['close'] >= last_data['vwap']].index.tolist()

		# Define breakout
		top_data = self.raw_data.loc[self.raw_data.index.get_level_values('code').isin(stocklist)]
		top_data['close_morethan_vwap'] = top_data['close'] >= top_data['vwap']
		top_data['breakout'] = top_data.groupby(level='code').rolling(window=2)['close_morethan_vwap']\
			.apply(lambda x: (x.iloc[0] == False) & (x.iloc[1] == True)).droplevel(0)
		
		# Get stockcodes with breakout
		stocklist = top_data['breakout'].groupby(level='code').any()
		stocklist = stocklist[stocklist].index.tolist()

		return stocklist

	async def _get_vwap_breakdown(self) -> list:
		"""Breakdown (t_x: close > vwap, t_y: close < vwap, within n days, and now close < vwap)"""
		# Get stockcodes with now close < vwap
		last_data = self.raw_data[['close','vwap']].groupby(level='code').last()
		stocklist = last_data[last_data['close'] <= last_data['vwap']].index.tolist()

		# Define breakdown
		top_data = self.raw_data.loc[self.raw_data.index.get_level_values('code').isin(stocklist)]
		top_data['close_lessthan_vwap'] = top_data['close'] <= top_data['vwap']
		top_data['breakdown'] = top_data.groupby(level='code').rolling(window=2)['close_lessthan_vwap']\
			.apply(lambda x: (x.iloc[0] == False) & (x.iloc[1] == True)).droplevel(0)
		
		# Get stockcodes with breakdown
		stocklist = top_data['breakdown'].groupby(level='code').any()
		stocklist = stocklist[stocklist].index.tolist()

		return stocklist