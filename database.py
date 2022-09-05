import os
# ==========
# DATABASE CONNECTION
# ==========
# Import Package
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Integer, String, Date, Numeric
from sqlalchemy.orm import Session

import os
env = os.getenv("ENV_PROD_DEV", "DEV")

# Create Database SQLAlchemy Engine
if env == "DEV":
	# PostgreSQL
	POSTGRES_DB_USER = os.getenv("POSTGRES_DB_USER")
	POSTGRES_DB_PASSWORD = os.getenv("POSTGRES_DB_PASSWORD")
	POSTGRES_DB_HOST = os.getenv("POSTGRES_DB_HOST")
	POSTGRES_DB_NAME = os.getenv("POSTGRES_DB_NAME")
	DB_URL = f"postgresql://{POSTGRES_DB_USER}:{POSTGRES_DB_PASSWORD}@{POSTGRES_DB_HOST}/{POSTGRES_DB_NAME}"

	# Create Engine
	engine = create_engine(DB_URL)

elif env == "PROD":
	# Bigquery
	BIGQUERY_PROJECT_ID = os.getenv("BIGQUERY_PROJECT_ID")
	BIGQUERY_DATASET_ID = os.getenv("BIGQUERY_DATASET_ID")
	BIGQUERY_LOCATION = os.getenv("BIGQUERY_LOCATION")
	BIGQUERY_DB_URL = f"bigquery://{BIGQUERY_PROJECT_ID}/{BIGQUERY_DATASET_ID}"

	BIGQUERY_CREDENTIALS_BASE64 = os.getenv("BIGQUERY_CREDENTIALS_BASE64")
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
	
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# SQLAlchemy DB Session Dependency
def get_dbs():
	dbs=SessionLocal()
	try:
		yield dbs
	finally:
		dbs.close()

# ==========
# SQLAlchemy Database Model
# ==========
class DataParam(Base):
	__tablename__ = "dataparam"
	index = Column(Integer, primary_key=True, autoincrement=True, index=True, nullable=False)
	param = Column(String)
	value = Column(String)

class ListBroker(Base):
	__tablename__ = "list_broker"
	index = Column(Integer, primary_key=True, autoincrement=True, index=True, nullable=False)
	code = Column(String(2), index=True, nullable=False)

class ListIndex(Base):
	__tablename__ = "list_index"
	index = Column(Integer, primary_key=True, autoincrement=True, index=True, nullable=False)
	code = Column(String(2), index=True, nullable=False)

class ListStock(Base):
	__tablename__ = "list_stock"
	index = Column(Integer, primary_key=True, autoincrement=True, index=True, nullable=False)
	code = Column(String, index=True, nullable=False)
	value = Column(Numeric)
	frequency = Column(Numeric)
	foreignsellval = Column(Numeric)
	foreignbuyval = Column(Numeric)

class IndexData(Base):
	__tablename__ = "indexdata"
	code = Column(String, index=True, nullable=False)
	date = Column(Date, index=True, nullable=False)
	previous = Column(Numeric)
	highest = Column(Numeric)
	lowest = Column(Numeric)
	close = Column(Numeric)
	numberofstock = Column(Numeric)
	change = Column(Numeric)
	volume = Column(Numeric)
	value = Column(Numeric)
	frequency = Column(Numeric)
	marketcapital = Column(Numeric)
	index = Column(Integer, primary_key=True, autoincrement=True, index=True, nullable=False)

class IndexTransactionCompositeBroker(Base):
	__tablename__ = "indextransaction_composite_broker"
	index = Column(Integer, primary_key=True, autoincrement=True, index=True, nullable=False)
	date = Column(Date, index=True, nullable=False)
	broker = Column(String(2), index=True, nullable=False)
	bfreq = Column(Numeric)
	bval = Column(Numeric)
	sfreq = Column(Numeric)
	sval = Column(Numeric)

class IndexTransactionCompositeForeign(Base):
	__tablename__ = "indextransaction_composite_foreign"
	index = Column(Integer, primary_key=True, autoincrement=True, index=True, nullable=False)
	date = Column(Date, index=True, nullable=False)
	foreignbuyval = Column(Numeric)
	foreignsellval = Column(Numeric)
	nonregularfrequency = Column(Numeric)
	nonregularvalue = Column(Numeric)

class StockData(Base):
	__tablename__ = "stockdata"
	code = Column(String, index=True, nullable=False)
	date = Column(Date, index=True, nullable=False)
	remarks = Column(String)
	previous = Column(Numeric)
	openprice = Column(Numeric)
	firsttrade = Column(Numeric)
	high = Column(Numeric)
	low = Column(Numeric)
	close = Column(Numeric)
	change = Column(Numeric)
	volume = Column(Numeric)
	value = Column(Numeric)
	frequency = Column(Numeric)
	indexindividual = Column(Numeric)
	offer = Column(Numeric)
	offervolume = Column(Numeric)
	bid = Column(Numeric)
	bidvolume = Column(Numeric)
	listedshares = Column(Numeric)
	tradebleshares = Column(Numeric)
	weightforindex = Column(Numeric)
	foreignsell = Column(Numeric)
	foreignbuy = Column(Numeric)
	delistingdate = Column(Date)
	nonregularvolume = Column(Numeric)
	nonregularvalue = Column(Numeric)
	nonregularfrequency = Column(Numeric)
	persen = Column(Numeric)
	percentage = Column(Numeric)
	index = Column(Integer, primary_key=True, autoincrement=True, index=True, nullable=False)

class StockTransaction(Base):
	__tablename__ = "stocktransaction"
	code = Column(String, index=True, nullable=False)
	date = Column(Date, index=True, nullable=False)
	broker = Column(String(2), index=True, nullable=False)
	bavg = Column(Numeric)
	bfreq = Column(Numeric)
	bval = Column(Numeric)
	bvol = Column(Numeric)
	savg = Column(Numeric)
	sfreq = Column(Numeric)
	sval = Column(Numeric)
	svol = Column(Numeric)
	index = Column(Integer, primary_key=True, autoincrement=True, index=True, nullable=False)

# INITIATE DATABASE
# Base.metadata.create_all(bind=engine)