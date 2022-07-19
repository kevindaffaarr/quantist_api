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

class FFFull(WhaleBase):
	def __init__(self,
		stockcode: str | None = None,
		startdate: datetime.date | None = None,
		enddate: datetime.date | None = datetime.date.today(),
		dbs: db.Session = next(db.get_dbs())
		):
		# Data Parameter
		self.stockcode = super()._get_default_stockcode(dbs) if stockcode is None else stockcode # Default stock is parameterized, may become a branding or endorsement option
		self.startdate = startdate # If the startdate is None, will be overridden by default_bar_range
		self.enddate = enddate # If the enddate is None, the default is already today date
		self.default_bar_range = super()._get_default_bar_range(dbs) if self.startdate is None else None # If the startdate is None, then the query goes to end date to the limit of default_bar_range
		
		# Raw Data
		self.raw_data = self.__get_raw_data(dbs,self.stockcode,self.startdate,self.enddate,self.default_bar_range)

	def __get_raw_data(self, dbs:db.Session = next(db.get_dbs()),
		stockcode: str = ...,
		startdate: datetime.date | None = None,
		enddate: datetime.date = ...,
		default_bar_range:int | None = None
		):
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
		
		return pd.read_sql(sql=qry.statement, con=dbs.bind, parse_dates=["date"])\
			.sort_values(by="date",ascending=True).reset_index(drop=True)
	
	