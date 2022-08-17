import pandas as pd
import numpy as np
import datetime
from sqlalchemy.sql import func
import database as db
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
		dbs: db.Session = next(db.get_dbs())
		):

		# Get defaults value
		default_ff = self.__get_default_ff(dbs)

		# Check Does Stock Code is composite
		self.stockcode = (str(default_ff['default_stockcode']) if stockcode is None else stockcode).lower() # Default stock is parameterized, may become a branding or endorsement option
		if self.stockcode == 'composite':
			self.type = 'composite'
		else:
			qry = dbs.query(db.ListStock.code).filter(db.ListStock.code == self.stockcode)
			row = pd.read_sql(sql=qry.statement, con=dbs.bind)
			if len(row) != 0:
				self.type = 'stock'
			else:
				raise KeyError("There is no such stock code in the database.")
		
		# Data Parameter
		self.startdate = startdate # If the startdate is None, will be overridden by default_bar_range
		self.enddate = enddate # If the enddate is None, the default is already today date
		default_bar_range = int(default_ff['default_bar_range']) if self.startdate is None else None # If the startdate is None, then the query goes to end date to the limit of default_bar_range
		period_fmf = int(default_ff['default_ff_period_fmf']) if period_fmf is None else period_fmf
		period_fprop = int(default_ff['default_ff_period_fprop']) if period_fprop is None else period_fprop
		period_fpricecorrel = int(default_ff['default_ff_period_fpricecorrel']) if period_fpricecorrel is None else period_fpricecorrel
		period_fmapricecorrel = int(default_ff['default_ff_period_fmapricecorrel']) if period_fmapricecorrel is None else period_fmapricecorrel
		period_fvwap = int(default_ff['default_ff_period_fvwap']) if period_fvwap is None else period_fvwap
		fpow_high_fprop = int(default_ff['default_ff_fpow_high_fprop']) if fpow_high_fprop is None else fpow_high_fprop
		fpow_high_fpricecorrel = int(default_ff['default_ff_fpow_high_fpricecorrel']) if fpow_high_fpricecorrel is None else fpow_high_fpricecorrel
		fpow_high_fmapricecorrel = int(default_ff['default_ff_fpow_high_fmapricecorrel']) if fpow_high_fmapricecorrel is None else fpow_high_fmapricecorrel
		fpow_medium_fprop = int(default_ff['default_ff_fpow_medium_fprop']) if fpow_medium_fprop is None else fpow_medium_fprop
		fpow_medium_fpricecorrel = int(default_ff['default_ff_fpow_medium_fpricecorrel']) if fpow_medium_fpricecorrel is None else fpow_medium_fpricecorrel
		fpow_medium_fmapricecorrel = int(default_ff['default_ff_fpow_medium_fmapricecorrel']) if fpow_medium_fmapricecorrel is None else fpow_medium_fmapricecorrel
		preoffset_period_param = max(period_fmf,period_fprop,period_fpricecorrel,(period_fmapricecorrel+period_fvwap))-1

		# Raw Data
		if self.type == 'stock':
			self.raw_data = self.__get_stock_raw_data(dbs,self.stockcode,\
				self.startdate,self.enddate,\
				default_bar_range,preoffset_period_param)
		elif self.type == 'composite':
			self.raw_data = self.__get_composite_raw_data(dbs,\
				self.startdate,self.enddate,\
				default_bar_range,preoffset_period_param)
			
		# Foreign Flow Indicators
		self.ff_indicators = self.calc_ff_indicators(self.raw_data,\
			period_fmf,period_fprop,period_fpricecorrel,period_fmapricecorrel,period_fvwap,\
			fpow_high_fprop,fpow_high_fpricecorrel,fpow_high_fmapricecorrel,fpow_medium_fprop,\
			fpow_medium_fpricecorrel,fpow_medium_fmapricecorrel,\
			preoffset_period_param
			)

		# Not to be ran inside init, but just as a method that return plotly fig
		# self.chart(media_type="json")
		
	def __get_default_ff(self, dbs:db.Session = next(db.get_dbs())) -> pd.Series:
		qry = dbs.query(db.DataParam.param, db.DataParam.value)\
			.filter((db.DataParam.param.like("default_ff_%")) | \
				(db.DataParam.param.like("default_stockcode")) | \
				(db.DataParam.param.like("default_bar_range")))
		return pd.Series(pd.read_sql(sql=qry.statement, con=dbs.bind).set_index("param")['value'])

	def __get_stock_raw_data(self, dbs:db.Session = next(db.get_dbs()),
		stockcode: str = ...,
		startdate: datetime.date | None = None,
		enddate: datetime.date = ...,
		default_bar_range:int | None = None,
		preoffset_period_param: int | None = 50
		) -> pd.DataFrame:
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
		if startdate is None:
			qry_main = qry.filter(db.StockData.date <= enddate)\
				.order_by(db.StockData.date.desc())\
				.limit(default_bar_range)
		else:
			qry_main = qry.filter(db.StockData.date.between(startdate, enddate))
		
		# Main Query Fetching
		raw_data_main = pd.read_sql(sql=qry_main.statement, con=dbs.bind, parse_dates=["date"])\
			.sort_values(by="date",ascending=True).reset_index(drop=True).set_index('date')
		
		# Pre-Data Query
		self.startdate = raw_data_main.index[0].date() # Assign self.startdate
		qry_pre = qry.filter(db.StockData.date < self.startdate)\
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

	def __get_composite_raw_data(self, dbs:db.Session = next(db.get_dbs()),
		startdate: datetime.date | None = None,
		enddate: datetime.date = ...,
		default_bar_range:int | None = None,
		preoffset_period_param: int | None = 50
		) -> pd.DataFrame:

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
		if startdate is None:
			qry_main = qry.filter(db.IndexData.date <= enddate)\
				.order_by(db.IndexData.date.desc())\
				.limit(default_bar_range)
		else:
			qry_main = qry.filter(db.IndexData.date.between(startdate, enddate))
		
		# Main Query Fetching
		raw_data_main = pd.read_sql(sql=qry_main.statement, con=dbs.bind, parse_dates=["date"])\
			.sort_values(by="date",ascending=True).reset_index(drop=True).set_index('date')
		
		# Pre-Data Query
		self.startdate = raw_data_main.index[0].date() # Assign self.startdate
		qry_pre = qry.filter(db.IndexData.date < self.startdate)\
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

	def calc_ff_indicators(
		self,
		raw_data: pd.DataFrame,
		period_fmf: int | None = 1,
		period_fprop: int | None = 1,
		period_fpricecorrel: int | None = 20,
		period_fmapricecorrel: int | None = 50,
		period_fvwap:int | None = 10,
		fpow_high_fprop: int | None = 40,
		fpow_high_fpricecorrel: int | None = 50,
		fpow_high_fmapricecorrel: int | None = 50,
		fpow_medium_fprop: int | None = 20,
		fpow_medium_fpricecorrel: int | None = 50,
		fpow_medium_fmapricecorrel: int | None = 50,
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
		raw_data['fnetprop'] = abs(raw_data['fbval']-raw_data['fsval']).rolling(window=period_fprop).sum()\
			/(raw_data['value'].rolling(window=period_fprop).sum()*2)

		# FPriceCorrel
		raw_data['fpricecorrel'] = raw_data['close'].rolling(window=period_fpricecorrel)\
			.corr(raw_data['netvol'])
		
		# FMAPriceCorrel
		raw_data['fmapricecorrel'] = raw_data['fpricecorrel'].rolling(window=period_fmapricecorrel).mean()
		
		# FVWAP
		raw_data['fvwap'] = (raw_data['netval'].rolling(window=period_fvwap).apply(lambda x: x[x>0].sum()))\
			/(raw_data['netvol'].rolling(window=period_fvwap).apply(lambda x: x[x>0].sum()))
		
		raw_data['fvwap'] = raw_data['fvwap'].mask(raw_data['fvwap'].le(0)).ffill()
		
		# FPow
		raw_data['fpow'] = np.where(
				(raw_data["fprop"]>(fpow_high_fprop/100)) & \
				(raw_data['fpricecorrel']>(fpow_high_fpricecorrel/100)) & \
				(raw_data['fmapricecorrel']>(fpow_high_fmapricecorrel/100)),3,
			np.where(
				(raw_data["fprop"]>(fpow_medium_fprop/100)) & \
				(raw_data['fpricecorrel']>(fpow_medium_fpricecorrel/100)) & \
				(raw_data['fmapricecorrel']>(fpow_medium_fmapricecorrel/100)),2,
				2
			)
		)

		# End of Method: Return Processed Raw Data to FF Indicators
		return raw_data.drop(raw_data.index[:preoffset_period_param])
	
	def chart(self,media_type:str | None = None):
		fig = genchart.foreign_chart(self.stockcode,self.ff_indicators)
		if media_type in ["png","jpeg","jpg","webp","svg"]:
			return genchart.fig_to_image(fig,media_type)
		elif media_type == "json":
			return genchart.fig_to_json(fig)
		else:
			return fig
	
class WhaleRadar():
	def __init__(self,
		startdate: datetime.date | None = None,
		enddate: datetime.date | None = datetime.date.today(),
		screener_min_value: int | None = None,
		screener_min_frequency: int | None = None,
		screener_min_fprop:int | None = None,
		stockcode_excludes: set[str] | None = set(),
		include_composite: bool | None = False,
		dbs: db.Session = next(db.get_dbs())
		):

		# Get default value of parameter
		default_radar = self.__get_default_radar()
		period_fmf = int(default_radar['default_radar_period_fmf']) if startdate is None else None
		period_fpricecorrel = int(default_radar['default_radar_period_fpricecorrel']) if startdate is None else None
		screener_min_value = int(default_radar['default_screener_min_value']) if screener_min_value is None else screener_min_value
		screener_min_frequency = int(default_radar['default_screener_min_frequency']) if screener_min_frequency is None else screener_min_frequency
		screener_min_fprop = int(default_radar['default_screener_min_fprop']) if screener_min_fprop is None else screener_min_fprop
		
		# Get filtered stockcodes
		filtered_stockcodes = self.__get_stockcodes(
			screener_min_value=screener_min_value,
			screener_min_frequency=screener_min_frequency,
			screener_min_fprop=screener_min_fprop,
			stockcode_excludes=stockcode_excludes,
			dbs=dbs)

		# Get raw data
		stocks_raw_data = self.__get_stocks_raw_data(
			startdate=startdate,
			enddate=enddate,
			filtered_stockcodes=filtered_stockcodes,
			bar_range=max(period_fmf,period_fpricecorrel) if startdate is None else None,
			dbs=dbs)

		# Calc Radar Indicators: last FMF
		# Calc Radar Indicators: last fpricecorrel OR last change_percentage

		pass
	
	def __get_default_radar(self, dbs:db.Session = next(db.get_dbs())) -> pd.Series:
		qry = dbs.query(db.DataParam.param, db.DataParam.value)\
			.filter((db.DataParam.param.like("default_radar_%")) | (db.DataParam.param.like("default_screener_%")))
		return pd.Series(pd.read_sql(sql=qry.statement, con=dbs.bind).set_index("param")['value'])
	
	def __get_stockcodes(self,
		screener_min_value: int | None = 5000000000,
		screener_min_frequency: int | None = 1000,
		screener_min_fprop:int | None = 0,
		stockcode_excludes: set[str] | None = set(),
		dbs: db.Session = next(db.get_dbs())
		) -> pd.Series:
		"""
		Get filtered stockcodes
		Filtered by:value>screener_min_value, frequency>screener_min_frequency 
					foreignbuyval>0, foreignsellval>0,
					stockcode_excludes
		"""
		# Query Definition
		qry = dbs.query(db.ListStock.code)\
			.filter((db.ListStock.value > screener_min_value) &
					(db.ListStock.frequency > screener_min_frequency) &
					(db.ListStock.foreignbuyval > 0) &
					(db.ListStock.foreignsellval > 0) &
					(((db.ListStock.foreignsellval+db.ListStock.foreignbuyval)/(db.ListStock.value*2)) > (screener_min_fprop/100)) &
					(db.ListStock.code.not_in(stockcode_excludes)))
		
		# Query Fetching: filtered_stockcodes
		return pd.Series(pd.read_sql(sql=qry.statement,con=dbs.bind).reset_index(drop=True)['code'])

	def __get_stocks_raw_data(self,
		startdate: datetime.date | None = None,
		enddate: datetime.date = ...,
		filtered_stockcodes:pd.Series = ...,
		bar_range: int | None = 5,
		dbs: db.Session = next(db.get_dbs())
		) -> pd.DataFrame:
		# Jika belum ada startdate, maka perlu ditentukan batas mulainya
		if startdate is None:
			# CARA 1 (0.028s)
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
			
			# CARA 2 (0.5s)
			# qry = dbs.query(db.StockData.code,
			# 	db.StockData.date,
			# 	db.StockData.close,
			# 	db.StockData.foreignbuy,
			# 	db.StockData.foreignsell,
			# 	func.row_number().over(
			# 		partition_by=db.StockData.code,
			# 		order_by=db.StockData.date.desc()
			# 	).label('no')
			# 	)\
			# 	.filter(db.StockData.code.in_(filtered_stockcodes.to_list()))\
			# 	.subquery()
			# qry = dbs.query(qry).filter(qry.c.no <= bar_range)
		
		# else:
		# 	startdate = startdate

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