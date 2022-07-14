# ==========
# DATABASE CONNECTION
# ==========
# Import Package
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from sqlalchemy import Column, Integer, String, Date, Numeric

from sqlalchemy.orm import Session
from sqlalchemy.sql import select

# PostgreSQL
DB_USER = "postgres"
DB_PASSWORD = "admin"
DB_HOST = "localhost"
DB_NAME = "quantist_marketdata"
DB_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"

# Create Engine
engine = create_engine(DB_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# DB Session Dependency
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
	volume = Column(Numeric)
	frequency = Column(Numeric)
	foreignsell = Column(Numeric)
	foreignbuy = Column(Numeric)

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
	bavg = Column(Numeric)
	bfreq = Column(Numeric)
	bval = Column(Numeric)
	bvol = Column(Numeric)
	savg = Column(Numeric)
	sfreq = Column(Numeric)
	sval = Column(Numeric)
	svol = Column(Numeric)

class IndexTransactionCompositeBroker(Base):
	__tablename__ = "indextransaction_composite_foreign"
	index = Column(Integer, primary_key=True, autoincrement=True, index=True, nullable=False)
	date = Column(Date, index=True, nullable=False)
	foreignbuy = Column(Numeric)
	foreignsell = Column(Numeric)
	nonregularfrequency = Column(Numeric)
	nonregularvalue = Column(Numeric)
	nonregularvolume = Column(Numeric)

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
# db.Base.metadata.create_all(bind=db.engine)