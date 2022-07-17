import pandas as pd
import datetime
import database as db

class FFFull:
	def __init__(self,
		stockcode: str | None = None,
		startdate: datetime.date | None = None,
		enddate: datetime.date | None = datetime.date.today(),
		dbs: db.Session = next(db.get_dbs())
	):
		# Data Parameter
		self.stockcode = self.__get_default_stockcode(dbs) if stockcode is None else stockcode
		__default_bar_range = self.__get_default_bar_range(dbs) if startdate is None else None

		# Raw Data
		self.raw_data = self.__get_raw_data(dbs,self.stockcode,startdate,enddate,__default_bar_range)

	def __get_default_stockcode(self, dbs:db.Session = next(db.get_dbs())):
		return dbs.query(db.DataParam.value)\
			.filter(db.DataParam.param == "default_stockcode")\
			.first()._asdict()['value']

	def __get_default_bar_range(self, dbs:db.Session = next(db.get_dbs())):
		return dbs.query(db.DataParam.value)\
			.filter(db.DataParam.param == "default_bar_range")\
			.first()._asdict()['value']
	
	def __get_raw_data(self, dbs:db.Session = next(db.get_dbs()),
		stockcode: str = ...,
		startdate: datetime.date | None = None,
		enddate: datetime.date = ...,
		default_bar_range = ...
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
			.sort_values(by="date",ascending=True)