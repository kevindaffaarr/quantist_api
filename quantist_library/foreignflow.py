from numpy import mask_indices
import pandas as pd
import datetime
import database as db

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

class StockFFFull(WhaleBase):
	def __init__(self,
		stockcode: str | None = None,
		startdate: datetime.date | None = None,
		enddate: datetime.date | None = datetime.date.today(),
		period_fmf: int | None = None,
		period_fprop: int | None = None,
		dbs: db.Session = next(db.get_dbs())
		):
		# Data Parameter
		self.stockcode = super()._get_default_stockcode(dbs) if stockcode is None else stockcode # Default stock is parameterized, may become a branding or endorsement option
		self.startdate = startdate # If the startdate is None, will be overridden by default_bar_range
		self.enddate = enddate # If the enddate is None, the default is already today date
		self.default_bar_range = super()._get_default_bar_range(dbs) if self.startdate is None else None # If the startdate is None, then the query goes to end date to the limit of default_bar_range
		self.period_fmf = period_fmf
		self.period_fprop = period_fprop

		# Raw Data: Assign variable self.raw_data
		self.__get_raw_data(dbs,self.stockcode,self.startdate,self.enddate,self.default_bar_range)

		# Foreign Flow Indicators
		self.ff_indicators = self.calc_ff_indicators(self.raw_data,self.period_fmf,self.period_fprop)

	def __get_raw_data(self, dbs:db.Session = next(db.get_dbs()),
		stockcode: str = ...,
		startdate: datetime.date | None = None,
		enddate: datetime.date = ...,
		default_bar_range:int | None = None
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
		if startdate is None:
			qry = qry.filter(db.StockData.date <= enddate)\
				.order_by(db.StockData.date.desc())\
				.limit(default_bar_range)
		else:
			qry = qry.filter(db.StockData.date.between(startdate, enddate))
		
		# Query Fetching
		raw_data = pd.read_sql(sql=qry.statement, con=dbs.bind, parse_dates=["date"])\
			.sort_values(by="date",ascending=True).reset_index(drop=True).set_index('date')

		# Data Cleansing: zero openprice replace with previous
		raw_data['openprice'] = raw_data['openprice'].mask(raw_data['openprice'].eq(0),raw_data['previous'])

		# End of Method: Return or Assign Attribute
		self.raw_data = raw_data

	def calc_ff_indicators(
		self,
		raw_data,
		period_fmf: int | None = 1,
		period_fprop: int | None =1,
		):
		# Define fbval, fsval, netvol, netval
		raw_data['fbval'] = raw_data['close']*raw_data['foreignbuy']
		raw_data['fsval'] = raw_data['close']*raw_data['foreignsell']
		raw_data['netvol'] = raw_data['foreignbuy']-raw_data['foreignsell']
		raw_data['netval'] = raw_data['fbval']-raw_data['fsval']
		
		# FF
		raw_data['ffvol'] = raw_data['netvol'].cumsum()

		# FMF
		raw_data['fmf'] = raw_data['netval'].rolling(window=period_fmf).sum()

		# FProp
		raw_data['fprop'] = (raw_data['fbval']+raw_data['fsval']).rolling(window=period_fprop).sum()\
			/(raw_data['value'].rolling(window=period_fprop).sum()*2)

		# FNetProp
		raw_data['fnetprop'] = abs(raw_data['fbval']-raw_data['fsval']).rolling(window=period_fprop).sum()\
			/(raw_data['value'].rolling(window=period_fprop).sum()*2)

		# FPriceCorrel
		# FMAPriceCorrel
		# FPow
		# FBuyAvg

		# End of Method: Return Processed Raw Data to FF Indicators
		ff_indicators = raw_data #.drop(...)
		return ff_indicators
		
		# TODO: Set limit bar range dilebihin (offset) sesuai dengan kebutuhan indicators period