import database as db
import pandas as pd

async def get_default_param() -> pd.Series:
	dbs: db.Session = next(db.get_dbs())
	qry = dbs.query(db.DataParam.param, db.DataParam.value)
	DEFAULT_PARAM = pd.Series(pd.read_sql(sql=qry.statement, con=dbs.bind).set_index("param")['value'])

	return DEFAULT_PARAM

async def get_list_stock() -> pd.DataFrame:
	dbs: db.Session = next(db.get_dbs())
	qry = dbs.query(db.ListStock.code)
	list_stock:pd.DataFrame = pd.read_sql(sql=qry.statement, con=dbs.bind)

	return list_stock