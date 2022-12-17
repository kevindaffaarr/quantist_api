# ==========
# DATABASE CONNECTION
# ==========
# Import Package
import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Integer, String, Date, Numeric
from sqlalchemy.orm import Session

env: str = os.getenv("ENV_PROD_DEV", "DEV")

# Create Database SQLAlchemy Engine
if env == "DEV":
	# PostgreSQL
	POSTGRES_DB_USER: str = os.getenv("POSTGRES_DB_USER","")
	POSTGRES_DB_PASSWORD: str = os.getenv("POSTGRES_DB_PASSWORD","")
	POSTGRES_DB_HOST: str = os.getenv("POSTGRES_DB_HOST","")
	POSTGRES_DB_NAME: str = os.getenv("POSTGRES_DB_NAME","")
	DB_URL: str = f"postgresql://{POSTGRES_DB_USER}:{POSTGRES_DB_PASSWORD}@{POSTGRES_DB_HOST}/{POSTGRES_DB_NAME}"

	# Create Engine
	engine = create_engine(DB_URL)

elif env == "PROD":
	# Bigquery
	BIGQUERY_PROJECT_ID: str = os.getenv("BIGQUERY_PROJECT_ID", "")
	BIGQUERY_DATASET_ID: str = os.getenv("BIGQUERY_DATASET_ID", "")
	BIGQUERY_LOCATION: str = os.getenv("BIGQUERY_LOCATION", "")
	BIGQUERY_DB_URL: str = f"bigquery://{BIGQUERY_PROJECT_ID}/{BIGQUERY_DATASET_ID}"

	BIGQUERY_CREDENTIALS_BASE64: str = os.getenv("BIGQUERY_CREDENTIALS_BASE64", "")
	# BIGQUERY_CREDENTIALS_JSON = json.loads(base64.b64decode(BIGQUERY_CREDENTIALS_BASE64))
	# BIGQUERY_CREDENTIALS = service_account.Credentials.from_service_account_info(BIGQUERY_CREDENTIALS_JSON)

	# Create SQLALCHEMY Engine
	engine = create_engine(
		url=BIGQUERY_DB_URL,
		credentials_base64=BIGQUERY_CREDENTIALS_BASE64,
		location=BIGQUERY_LOCATION,
		)

else:
	raise ValueError("ENV_PROD_DEV must be DEV or PROD")
	
SessionLocal: sessionmaker = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base(bind=engine)

# SQLAlchemy DB Session Dependency
def get_dbs():
	dbs: Session=SessionLocal()
	try:
		yield dbs
	finally:
		dbs.close()

# ==========
# SQLAlchemy Database Model
# ==========
class DataParam(Base):
	__tablename__: str = "dataparam"
	index: Column = Column(Integer, primary_key=True, autoincrement=True, index=True, nullable=False)
	param: Column = Column(String)
	value: Column = Column(String)

class ListBroker(Base):
	__tablename__: str = "list_broker"
	index: Column = Column(Integer, primary_key=True, autoincrement=True, index=True, nullable=False)
	code: Column = Column(String(2), index=True, nullable=False)

class ListIndex(Base):
	__tablename__: str = "list_index"
	index: Column = Column(Integer, primary_key=True, autoincrement=True, index=True, nullable=False)
	code: Column = Column(String(2), index=True, nullable=False)

class ListStock(Base):
	__tablename__: str = "list_stock"
	index: Column = Column(Integer, primary_key=True, autoincrement=True, index=True, nullable=False)
	code: Column = Column(String, index=True, nullable=False)
	value: Column = Column(Numeric)
	frequency: Column = Column(Numeric)
	foreignsellval: Column = Column(Numeric)
	foreignbuyval: Column = Column(Numeric)

class IndexData(Base):
	__tablename__: str = "indexdata"
	code: Column = Column(String, index=True, nullable=False)
	date: Column = Column(Date, index=True, nullable=False)
	previous: Column = Column(Numeric)
	highest: Column = Column(Numeric)
	lowest: Column = Column(Numeric)
	close: Column = Column(Numeric)
	numberofstock: Column = Column(Numeric)
	change: Column = Column(Numeric)
	volume: Column = Column(Numeric)
	value: Column = Column(Numeric)
	frequency: Column = Column(Numeric)
	marketcapital: Column = Column(Numeric)
	index: Column = Column(Integer, primary_key=True, autoincrement=True, index=True, nullable=False)

class IndexTransactionCompositeBroker(Base):
	__tablename__: str = "indextransaction_composite_broker"
	index: Column = Column(Integer, primary_key=True, autoincrement=True, index=True, nullable=False)
	date: Column = Column(Date, index=True, nullable=False)
	broker: Column = Column(String(2), index=True, nullable=False)
	bfreq: Column = Column(Numeric)
	bval: Column = Column(Numeric)
	sfreq: Column = Column(Numeric)
	sval: Column = Column(Numeric)

class IndexTransactionCompositeForeign(Base):
	__tablename__: str = "indextransaction_composite_foreign"
	index: Column = Column(Integer, primary_key=True, autoincrement=True, index=True, nullable=False)
	date: Column = Column(Date, index=True, nullable=False)
	foreignbuyval: Column = Column(Numeric)
	foreignsellval: Column = Column(Numeric)
	nonregularfrequency: Column = Column(Numeric)
	nonregularvalue: Column = Column(Numeric)

class StockData(Base):
	__tablename__: str = "stockdata"
	code: Column = Column(String, index=True, nullable=False)
	date: Column = Column(Date, index=True, nullable=False)
	remarks: Column = Column(String)
	previous: Column = Column(Numeric)
	openprice: Column = Column(Numeric)
	firsttrade: Column = Column(Numeric)
	high: Column = Column(Numeric)
	low: Column = Column(Numeric)
	close: Column = Column(Numeric)
	change: Column = Column(Numeric)
	volume: Column = Column(Numeric)
	value: Column = Column(Numeric)
	frequency: Column = Column(Numeric)
	indexindividual: Column = Column(Numeric)
	offer: Column = Column(Numeric)
	offervolume: Column = Column(Numeric)
	bid: Column = Column(Numeric)
	bidvolume: Column = Column(Numeric)
	listedshares: Column = Column(Numeric)
	tradebleshares: Column = Column(Numeric)
	weightforindex: Column = Column(Numeric)
	foreignsell: Column = Column(Numeric)
	foreignbuy: Column = Column(Numeric)
	delistingdate: Column = Column(Date)
	nonregularvolume: Column = Column(Numeric)
	nonregularvalue: Column = Column(Numeric)
	nonregularfrequency: Column = Column(Numeric)
	persen: Column = Column(Numeric)
	percentage: Column = Column(Numeric)
	index: Column = Column(Integer, primary_key=True, autoincrement=True, index=True, nullable=False)

class StockTransaction(Base):
	__tablename__: str = "stocktransaction"
	code: Column = Column(String, index=True, nullable=False)
	date: Column = Column(Date, index=True, nullable=False)
	broker: Column = Column(String(2), index=True, nullable=False)
	bavg: Column = Column(Numeric)
	bfreq: Column = Column(Numeric)
	bval: Column = Column(Numeric)
	bvol: Column = Column(Numeric)
	savg: Column = Column(Numeric)
	sfreq: Column = Column(Numeric)
	sval: Column = Column(Numeric)
	svol: Column = Column(Numeric)
	index: Column = Column(Integer, primary_key=True, autoincrement=True, index=True, nullable=False)

class ScreenerResult(Base):
	__tablename__: str = "screener_result"
	date: Column = Column(Date, index=True, nullable=False)
	screener_method: Column = Column(String, index=True, nullable=False)
	code: Column = Column(String, index=True, nullable=False)
	close: Column = Column(Numeric)
	money_flow: Column = Column(Numeric)
	proportion: Column = Column(Numeric)
	correlation: Column = Column(Numeric)
	vwap: Column = Column(Numeric)
	index: Column = Column(Integer, primary_key=True, autoincrement=True, index=True, nullable=False)

# INITIATE DATABASE
# Base.metadata.create_all(bind=engine)
