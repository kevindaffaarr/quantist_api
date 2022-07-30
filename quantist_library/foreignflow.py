import pandas as pd
import numpy as np
import datetime
import database as db
from quantist_library import genchart

class WhaleBase():
	def __init__(self) -> None:
		pass

	def _get_default_stockcode(self, dbs:db.Session = next(db.get_dbs())):
		return dbs.query(db.DataParam.value)\
			.filter(db.DataParam.param == "default_stockcode")\
			.first()._asdict()['value']

	def _get_default_bar_range(self, dbs:db.Session = next(db.get_dbs())):
		return dbs.query(db.DataParam.value)\
			.filter(db.DataParam.param == "default_bar_range")\
			.first()._asdict()['value']

	def _get_default_ff(self, dbs:db.Session = next(db.get_dbs())):
		qry = dbs.query(db.DataParam.param, db.DataParam.value)\
			.filter(db.DataParam.param.like("default_ff_%"))
		return pd.Series(\
			pd.read_sql(sql=qry.statement, con=dbs.bind).set_index("param")['value']\
			).astype(int)

class StockFFFull(WhaleBase):
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

		# Check Does Stock Code Exists
		self.stockcode = (super()._get_default_stockcode(dbs) if stockcode is None else stockcode).lower() # Default stock is parameterized, may become a branding or endorsement option
		qry = dbs.query(db.ListStock.code).filter(db.ListStock.code == self.stockcode)
		row = pd.read_sql(sql=qry.statement, con=dbs.bind)
		
		if len(row) == 0:
			raise KeyError("There is no such stock code in the database.")
		else:
			# Data Parameter
			self.startdate = startdate # If the startdate is None, will be overridden by default_bar_range
			self.enddate = enddate # If the enddate is None, the default is already today date
			default_bar_range = super()._get_default_bar_range(dbs) if self.startdate is None else None # If the startdate is None, then the query goes to end date to the limit of default_bar_range
			default_ff = super()._get_default_ff(dbs)
			period_fmf = default_ff['default_ff_period_fmf'] if period_fmf is None else period_fmf
			period_fprop = default_ff['default_ff_period_fprop'] if period_fprop is None else period_fprop
			period_fpricecorrel = default_ff['default_ff_period_fpricecorrel'] if period_fpricecorrel is None else period_fpricecorrel
			period_fmapricecorrel = default_ff['default_ff_period_fmapricecorrel'] if period_fmapricecorrel is None else period_fmapricecorrel
			period_fvwap = default_ff['default_ff_period_fvwap'] if period_fvwap is None else period_fvwap
			fpow_high_fprop = default_ff['default_ff_fpow_high_fprop'] if fpow_high_fprop is None else fpow_high_fprop
			fpow_high_fpricecorrel = default_ff['default_ff_fpow_high_fpricecorrel'] if fpow_high_fpricecorrel is None else fpow_high_fpricecorrel
			fpow_high_fmapricecorrel = default_ff['default_ff_fpow_high_fmapricecorrel'] if fpow_high_fmapricecorrel is None else fpow_high_fmapricecorrel
			fpow_medium_fprop = default_ff['default_ff_fpow_medium_fprop'] if fpow_medium_fprop is None else fpow_medium_fprop
			fpow_medium_fpricecorrel = default_ff['default_ff_fpow_medium_fpricecorrel'] if fpow_medium_fpricecorrel is None else fpow_medium_fpricecorrel
			fpow_medium_fmapricecorrel = default_ff['default_ff_fpow_medium_fmapricecorrel'] if fpow_medium_fmapricecorrel is None else fpow_medium_fmapricecorrel
			preoffset_period_param = max(period_fmf,period_fprop,period_fpricecorrel,(period_fmapricecorrel+period_fvwap))-1

			# Raw Data: Assign variable self.raw_data
			self.__get_raw_data(dbs,self.stockcode,\
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
			# self.chart()
		
	def __get_raw_data(self, dbs:db.Session = next(db.get_dbs()),
		stockcode: str = ...,
		startdate: datetime.date | None = None,
		enddate: datetime.date = ...,
		default_bar_range:int | None = None,
		preoffset_period_param: int | None = 50
		):
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
		self.raw_data = raw_data

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
		):
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
	
	def chart(self):
		fig = genchart.foreign_chart(self.stockcode,self.ff_indicators)
		return fig