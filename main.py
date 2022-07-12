from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy.sql import select

from routers import whalechart

import database, dependencies
database.Base.metadata.create_all(bind=database.engine)

"""
=============================
FAST API APP
Start the server by run this code in terminal
uvicorn main:app --reload
=============================
"""

"""
=============================
META DATA:
=============================
"""
description = """
	## Backend of Quantist.io: Capturing The Silhouette of Data
	_Democratize the information of Indonesia Stocks and more to investors from retail to institution._

	Consists of high-end analysis tools based on data with top-down analysis:
	* Conservation of money (forex/crypto - bond - stocks)
	* External factor (commodities, national/international issues, sentiments)
	* Fundamental: business performance, corporate action
	* Transaction: broker summary, done trade, 5% shares holder, share holder distribution
	* Behaviour: supply & demand, momentum, trend, time
"""

# INITIATE APP
app = FastAPI(
	title="quantist_api",
	description=description,
	version="0.0.1",
	contact={
		"name": "Kevin Daffa Arrahman",
		"email": "kevindaffaarr@quantist.io"
	}
)

# CORS https://fastapi.tiangolo.com/tutorial/cors/
origins = ["*"]
app.add_middleware(
	CORSMiddleware,
	allow_origins=origins,
	allow_credentials=True,
	allow_methods=["GET","POST","PUT"],
	allow_headers=["*"]
)

# DB Dependency
def get_db():
	db=database.SessionLocal()
	try:
		yield db
	finally:
		db.close()

# Function
def get_list_stock(db: Session, only_code: bool = True):
	if only_code:
		return db.execute(select(database.ListStock.index,database.ListStock.code)).scalars().all()
	else:
		return db.execute(select(database.ListStock)).scalars().all()

# INCLUDE ROUTER
app.include_router(whalechart.router)

@app.get("/")
async def home():
	return {"message": "Welcome to Quantist.io"}

@app.get("/list/{list_category}", response_model=list[dependencies.ListStockExtended])
def get_list(list_category: dependencies.ListCategory, db: Session = Depends(get_db)):
	db_list_stock = get_list_stock(db=db, only_code=False)
	return db_list_stock