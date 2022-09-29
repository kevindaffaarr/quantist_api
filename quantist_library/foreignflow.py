from __future__ import annotations
import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
from sqlalchemy.sql import func
import database as db
import dependencies as dp
from quantist_library import genchart

class StockFFFull():
	def __init__(self,
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
		dbs: db.Session | None = next(db.get_dbs())
		) -> None:
		self.stockcode = stockcode
		self.startdate = startdate
		self.enddate = enddate
		self.period_fmf = period_fmf
		self.period_fprop = period_fprop
		self.period_fpricecorrel = period_fpricecorrel
		self.period_fmapricecorrel = period_fmapricecorrel
		self.period_fvwap = period_fvwap
		self.fpow_high_fprop = fpow_high_fprop
		self.fpow_high_fpricecorrel = fpow_high_fpricecorrel
		self.fpow_high_fmapricecorrel = fpow_high_fmapricecorrel
		self.fpow_medium_fprop = fpow_medium_fprop
		self.fpow_medium_fpricecorrel = fpow_medium_fpricecorrel
		self.fpow_medium_fmapricecorrel = fpow_medium_fmapricecorrel
		self.dbs = dbs

		self.type = None
		self.ff_indicators = pd.DataFrame()

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
		self.period_fmf = int(default_ff['default_ff_period_fmf']) if self.period_fmf is None else self.period_fmf
		self.period_fprop = int(default_ff['default_ff_period_fprop']) if self.period_fprop is None else self.period_fprop
		self.period_fpricecorrel = int(default_ff['default_ff_period_fpricecorrel']) if self.period_fpricecorrel is None else self.period_fpricecorrel
		self.period_fmapricecorrel = int(default_ff['default_ff_period_fmapricecorrel']) if self.period_fmapricecorrel is None else self.period_fmapricecorrel
		self.period_fvwap = int(default_ff['default_ff_period_fvwap']) if self.period_fvwap is None else self.period_fvwap
		self.fpow_high_fprop = int(default_ff['default_ff_fpow_high_fprop']) if self.fpow_high_fprop is None else self.fpow_high_fprop
		self.fpow_high_fpricecorrel = int(default_ff['default_ff_fpow_high_fpricecorrel']) if self.fpow_high_fpricecorrel is None else self.fpow_high_fpricecorrel
		self.fpow_high_fmapricecorrel = int(default_ff['default_ff_fpow_high_fmapricecorrel']) if self.fpow_high_fmapricecorrel is None else self.fpow_high_fmapricecorrel
		self.fpow_medium_fprop = int(default_ff['default_ff_fpow_medium_fprop']) if self.fpow_medium_fprop is None else self.fpow_medium_fprop
		self.fpow_medium_fpricecorrel = int(default_ff['default_ff_fpow_medium_fpricecorrel']) if self.fpow_medium_fpricecorrel is None else self.fpow_medium_fpricecorrel
		self.fpow_medium_fmapricecorrel = int(default_ff['default_ff_fpow_medium_fmapricecorrel']) if self.fpow_medium_fmapricecorrel is None else self.fpow_medium_fmapricecorrel
		preoffset_period_param = max(self.period_fmf,self.period_fprop,self.period_fpricecorrel,(self.period_fmapricecorrel+self.period_fvwap))-1

		# Raw Data
		if self.type == 'stock':
			raw_data = await self.__get_stock_raw_data(self.dbs,self.stockcode,\
				self.startdate,self.enddate,\
				default_months_range,preoffset_period_param)
		elif self.type == 'composite':
			raw_data = await self.__get_composite_raw_data(self.dbs,\
				self.startdate,self.enddate,\
				default_months_range,preoffset_period_param)
			
		# Foreign Flow Indicators
		self.ff_indicators = await self.calc_ff_indicators(raw_data,\
			self.period_fmf,self.period_fprop,self.period_fpricecorrel,self.period_fmapricecorrel,self.period_fvwap,\
			self.fpow_high_fprop,self.fpow_high_fpricecorrel,self.fpow_high_fmapricecorrel,self.fpow_medium_fprop,\
			self.fpow_medium_fpricecorrel,self.fpow_medium_fmapricecorrel,\
			preoffset_period_param
			)

		# Not to be ran inside init, but just as a method that return plotly fig
		# self.chart(media_type="json")

		return self
		
	async def __get_default_ff(self, dbs:db.Session | None = next(db.get_dbs())) -> pd.Series:
		qry = dbs.query(db.DataParam.param, db.DataParam.value)\
			.filter((db.DataParam.param.like("default_ff_%")) | \
				(db.DataParam.param.like("default_stockcode")) | \
				(db.DataParam.param.like("default_months_range")))
		return pd.Series(pd.read_sql(sql=qry.statement, con=dbs.bind).set_index("param")['value'])

	async def __get_stock_raw_data(self,
		dbs:db.Session | None = next(db.get_dbs()),
		stockcode: str = ...,
		startdate: datetime.date | None = None,
		enddate: datetime.date = datetime.date.today(),
		default_months_range:int | None = None,
		preoffset_period_param: int | None = 50
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
		qry_main = qry.filter(db.StockData.date.between(startdate, enddate))
		
		# Main Query Fetching
		raw_data_main = pd.read_sql(sql=qry_main.statement, con=dbs.bind, parse_dates=["date"])\
			.sort_values(by="date",ascending=True).reset_index(drop=True).set_index('date')
		
		# Pre-Data Query
		startdate = raw_data_main.index[0].date() # Assign startdate to new defined startdate
		self.startdate = raw_data_main.index[0].date() # Assign self.startdate to new defined startdate
		qry_pre = qry.filter(db.StockData.date < startdate)\
			.order_by(db.StockData.date.desc())\
			.limit(preoffset_period_param)

		# Pre-Data Query Fetching
		raw_data_pre = pd.read_sql(sql=qry_pre.statement, con=dbs.bind, parse_dates=["date"])\
			.sort_values(by="date",ascending=True).reset_index(drop=True).set_index('date')

		# Concatenate Pre and Main Query
		raw_data = pd.concat([raw_data_pre,raw_data_main])

		# Data Cleansing: zero openprice replace with previous
		raw_data['openprice'] = raw_data['openprice'].mask(raw_data['openprice'].eq(0),raw_data['previous'])

		# End of Method: Return or Assign Attribute
		return raw_data

	async def __get_composite_raw_data(self, 
		dbs:db.Session | None = next(db.get_dbs()),
		startdate: datetime.date | None = None,
		enddate: datetime.date = datetime.date.today(),
		default_months_range:int | None = None,
		preoffset_period_param: int | None = 50
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
		qry_main = qry.filter(db.IndexData.date.between(startdate, enddate))
		
		# Main Query Fetching
		raw_data_main = pd.read_sql(sql=qry_main.statement, con=dbs.bind, parse_dates=["date"])\
			.sort_values(by="date",ascending=True).reset_index(drop=True).set_index('date')
		
		# Pre-Data Query
		startdate = raw_data_main.index[0].date() # Assign startdate to new defined startdate
		self.startdate = raw_data_main.index[0].date() # Assign self.startdate
		qry_pre = qry.filter(db.IndexData.date < startdate)\
			.order_by(db.IndexData.date.desc())\
			.limit(preoffset_period_param)

		# Pre-Data Query Fetching
		raw_data_pre = pd.read_sql(sql=qry_pre.statement, con=dbs.bind, parse_dates=["date"])\
			.sort_values(by="date",ascending=True).reset_index(drop=True).set_index('date')

		# Concatenate Pre and Main Query
		raw_data = pd.concat([raw_data_pre,raw_data_main])

		# Convert Foreign Transaction Value to Volume
		raw_data["foreignbuy"] = raw_data["foreignbuy"]/raw_data["close"]
		raw_data["foreignsell"] = raw_data["foreignsell"]/raw_data["close"]

		# End of Method: Return or Assign Attribute
		return raw_data

	async def calc_ff_indicators(self,
		raw_data: pd.DataFrame,
		period_fmf: int | None = 1,
		period_fprop: int | None = 10,
		period_fpricecorrel: int | None = 10,
		period_fmapricecorrel: int | None = 100,
		period_fvwap:int | None = 21,
		fpow_high_fprop: int | None = 40,
		fpow_high_fpricecorrel: int | None = 50,
		fpow_high_fmapricecorrel: int | None = 30,
		fpow_medium_fprop: int | None = 20,
		fpow_medium_fpricecorrel: int | None = 30,
		fpow_medium_fmapricecorrel: int | None = 30,
		preoffset_period_param: int | None = 0
		) -> pd.DataFrame:
		# Define fbval, fsval, netvol, netval
		raw_data['fbval'] = raw_data['close']*raw_data['foreignbuy']
		raw_data['fsval'] = raw_data['close']*raw_data['foreignsell']
		raw_data['netvol'] = raw_data['foreignbuy']-raw_data['foreignsell']
		raw_data['netval'] = raw_data['fbval']-raw_data['fsval']
		
		# FF
		raw_data['fvolflow'] = raw_data['netvol'].cumsum()

		# FMF
		raw_data['fmf'] = raw_data['netval'].rolling(window=period_fmf).sum()

		# FProp
		raw_data['fprop'] = (raw_data['fbval']+raw_data['fsval']).rolling(window=period_fprop).sum()\
			/(raw_data['value'].rolling(window=period_fprop).sum()*2)

		# FNetProp
		raw_data['fnetprop'] = abs(raw_data['netval']).rolling(window=period_fprop).sum()\
			/(raw_data['value'].rolling(window=period_fprop).sum()*2)

		# FPriceCorrel
		raw_data['fpricecorrel'] = raw_data['close'].rolling(window=period_fpricecorrel)\
			.corr(raw_data['fvolflow'])
		
		# FMAPriceCorrel
		raw_data['fmapricecorrel'] = raw_data['fpricecorrel'].rolling(window=period_fmapricecorrel).mean()
		
		# FVWAP
		raw_data['fvwap'] = (raw_data['netval'].rolling(window=period_fvwap).apply(lambda x: x[x>0].sum()))\
			/(raw_data['netvol'].rolling(window=period_fvwap).apply(lambda x: x[x>0].sum()))
		
		raw_data['fvwap'] = raw_data['fvwap'].mask(raw_data['fvwap'].le(0)).ffill()
		
		# FPow
		raw_data['fpow'] = \
			np.where(
				(raw_data["fprop"]>(fpow_high_fprop/100)) & \
				(raw_data['fpricecorrel']>(fpow_high_fpricecorrel/100)) & \
				(raw_data['fmapricecorrel']>(fpow_high_fmapricecorrel/100)), \
				3,
				np.where(
					(raw_data["fprop"]>(fpow_medium_fprop/100)) & \
					(raw_data['fpricecorrel']>(fpow_medium_fpricecorrel/100)) & \
					(raw_data['fmapricecorrel']>(fpow_medium_fmapricecorrel/100)), \
					2, \
					1
				)
			)

		# End of Method: Return Processed Raw Data to FF Indicators
		return raw_data.drop(raw_data.index[:preoffset_period_param])
	
	async def chart(self,media_type:str | None = None):
		fig = await genchart.foreign_chart(self.stockcode,self.ff_indicators)
		if media_type in ["png","jpeg","jpg","webp","svg"]:
			return await genchart.fig_to_image(fig,media_type)
		elif media_type == "json":
			return await genchart.fig_to_json(fig)
		else:
			return fig
	
class ForeignRadar():
	def __init__(self,
		startdate: datetime.date | None = None,
		enddate: datetime.date | None = datetime.date.today(),
		y_axis_type: dp.ListRadarType | None = "correlation",
		stockcode_excludes: set[str] | None = set(),
		include_composite: bool | None = False,
		screener_min_value: int | None = None,
		screener_min_frequency: int | None = None,
		screener_min_fprop:int | None = None,
		period_fmf: int | None = None,
		period_fpricecorrel: int | None = None,
		dbs: db.Session | None = next(db.get_dbs())
		) -> None:
		self.startdate = startdate
		self.enddate = enddate
		self.y_axis_type = y_axis_type
		self.stockcode_excludes = stockcode_excludes
		self.include_composite = include_composite
		self.screener_min_value = screener_min_value
		self.screener_min_frequency = screener_min_frequency
		self.screener_min_fprop = screener_min_fprop
		self.period_fmf = period_fmf
		self.period_fpricecorrel = period_fpricecorrel
		self.dbs = dbs

		self.radar_indicators =  pd.DataFrame()

	async def fit(self) -> ForeignRadar:
		# Get default value of parameter
		default_radar = await self.__get_default_radar()
		self.period_fmf = int(default_radar['default_radar_period_fmf']) if self.startdate is None else None
		self.period_fpricecorrel = int(default_radar['default_radar_period_fpricecorrel']) if self.startdate is None else None
		self.screener_min_value = int(default_radar['default_screener_min_value']) if self.screener_min_value is None else self.screener_min_value
		self.screener_min_frequency = int(default_radar['default_screener_min_frequency']) if self.screener_min_frequency is None else self.screener_min_frequency
		self.screener_min_fprop = int(default_radar['default_screener_min_fprop']) if self.screener_min_fprop is None else self.screener_min_fprop
		
		# Get filtered stockcodes
		filtered_stockcodes = await self.__get_stockcodes(
			screener_min_value=self.screener_min_value,
			screener_min_frequency=self.screener_min_frequency,
			screener_min_fprop=self.screener_min_fprop,
			stockcode_excludes=self.stockcode_excludes,
			dbs=self.dbs)

		# Get raw data
		stocks_raw_data = await self.__get_stocks_raw_data(
			startdate=self.startdate,
			enddate=self.enddate,
			filtered_stockcodes=filtered_stockcodes,
			bar_range=max(self.period_fmf,self.period_fpricecorrel) if self.startdate is None else None,
			dbs=self.dbs)

		# Set startdate and enddate based on data availability
		self.startdate:datetime.date = stocks_raw_data['date'].min().date()
		self.enddate:datetime.date = stocks_raw_data['date'].max().date()

		# Calc Radar Indicators: last fpricecorrel OR last changepercentage
		self.radar_indicators = await self.calc_radar_indicators(\
			stocks_raw_data=stocks_raw_data,y_axis_type=self.y_axis_type)

		if self.include_composite is True:
			composite_raw_data = await self.__get_composite_raw_data(
				startdate=self.startdate,
				enddate=self.enddate,
				bar_range=max(self.period_fmf,self.period_fpricecorrel) if self.startdate is None else None,
				dbs=self.dbs
			)
			composite_radar_indicators = await self.calc_radar_indicators(\
				stocks_raw_data=composite_raw_data,y_axis_type=self.y_axis_type)
			
			self.radar_indicators = pd.concat([self.radar_indicators,composite_radar_indicators])

		# Not to be ran inside init, but just as a method that return plotly fig
		# self.chart(media_type="json")

		return self
		
	async def __get_default_radar(self, dbs:db.Session | None = next(db.get_dbs())) -> pd.Series:
		qry = dbs.query(db.DataParam.param, db.DataParam.value)\
			.filter((db.DataParam.param.like("default_radar_%")) | (db.DataParam.param.like("default_screener_%")))
		return pd.Series(pd.read_sql(sql=qry.statement, con=dbs.bind).set_index("param")['value'])
	
	async def __get_stockcodes(self,
		screener_min_value: int | None = 5000000000,
		screener_min_frequency: int | None = 1000,
		screener_min_fprop:int | None = 0,
		stockcode_excludes: set[str] | None = set(),
		dbs: db.Session | None = next(db.get_dbs())
		) -> pd.Series:
		"""
		Get filtered stockcodes
		Filtered by:value>screener_min_value, frequency>screener_min_frequency 
					foreignbuyval>0, foreignsellval>0,
					stockcode_excludes
		"""
		# Query Definition
		stockcode_excludes_lower = set(x.lower() for x in stockcode_excludes)
		qry = dbs.query(db.ListStock.code)\
			.filter((db.ListStock.value > screener_min_value) &
					(db.ListStock.frequency > screener_min_frequency) &
					(db.ListStock.foreignbuyval > 0) &
					(db.ListStock.foreignsellval > 0) &
					(((db.ListStock.foreignsellval+db.ListStock.foreignbuyval)/(db.ListStock.value*2)) > (screener_min_fprop/100)) &
					(db.ListStock.code.not_in(stockcode_excludes_lower)))
		
		# Query Fetching: filtered_stockcodes
		return pd.Series(pd.read_sql(sql=qry.statement,con=dbs.bind).reset_index(drop=True)['code'])

	async def __get_stocks_raw_data(self,
		startdate: datetime.date | None = None,
		enddate: datetime.date = datetime.date.today(),
		filtered_stockcodes:pd.Series = ...,
		bar_range: int | None = 5,
		dbs: db.Session | None = next(db.get_dbs())
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
			.filter(db.StockData.date.between(startdate,enddate))
		
		# Query Fetching: stocks raw data
		stocks_raw_data = pd.read_sql(sql=qry.statement,con=dbs.bind,parse_dates=["date"])\
			.sort_values(by=["code","date"],ascending=[True,True]).reset_index(drop=True)
		return stocks_raw_data
	
	async def __get_composite_raw_data(self,
		startdate:datetime.date|None=None,
		enddate:datetime.date|None=None,
		bar_range:int|None=5,
		dbs:db.Session | None = next(db.get_dbs())
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
			(db.IndexTransactionCompositeForeign.foreignbuyval/db.IndexData.close).label('foreignbuy'),
			(db.IndexTransactionCompositeForeign.foreignsellval/db.IndexData.close).label('foreignsell')
			).\
			filter(db.IndexData.code=="composite")\
			.filter(db.IndexData.date.between(startdate,enddate))\
			.join(db.IndexTransactionCompositeForeign,
				db.IndexTransactionCompositeForeign.date == db.IndexData.date
			)

		# Query Fetching: composite raw data
		composite_raw_data = pd.read_sql(sql=qry.statement,con=dbs.bind,parse_dates=["date"])\
			.sort_values(by=["date"],ascending=[True]).reset_index(drop=True)
		return composite_raw_data

	async def calc_radar_indicators(self,
		stocks_raw_data:pd.DataFrame,
		y_axis_type:dp.ListRadarType | None = "correlation"
		) -> pd.DataFrame:
		
		radar_indicators = pd.DataFrame()

		# Y axis: fmf
		stocks_raw_data['netval'] = stocks_raw_data['close']*\
			(stocks_raw_data['foreignbuy']-stocks_raw_data['foreignsell'])
		radar_indicators['mf'] = stocks_raw_data.groupby(by='code')['netval'].sum()

		# X axis:
		if y_axis_type == "correlation":
			# NetVol
			stocks_raw_data['netvol'] = stocks_raw_data['foreignbuy']-stocks_raw_data['foreignsell']
			# FF
			stocks_raw_data['fvolflow'] = stocks_raw_data.groupby('code')['netvol'].cumsum()
			# FPriceCorrel
			radar_indicators[y_axis_type] = stocks_raw_data.groupby(by='code')['fvolflow']\
				.corr(stocks_raw_data['close'])
				
		elif y_axis_type == "changepercentage":
			radar_indicators[y_axis_type] = \
				(stocks_raw_data.groupby(by='code')['close'].nth([-1])- \
				stocks_raw_data.groupby(by='code')['close'].nth([0]))/ \
				stocks_raw_data.groupby(by='code')['close'].nth([0])
		else:
			raise Exception("Not a valid radar type")
		
		return radar_indicators
	
	async def chart(self,media_type:str | None = None):
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
