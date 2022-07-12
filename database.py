# ==========
# DATABASE CONNECTION
# ==========
# Import Package
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from sqlalchemy import Column, Integer, String, Date, Numeric
from sqlalchemy.orm import relationship

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

# Database Model
class ListStock(Base):
	__tablename__ = "list_stock"
	index = Column(Integer, primary_key=True)
	code = Column(String)
	volume = Column(Numeric)
	frequency = Column(Numeric)
	foreignsell = Column(Numeric)
	foreignbuy = Column(Numeric)