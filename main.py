import os
import warnings
# Ignore FutureWarning, DeprecationWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

ENV_OR_PROD = os.getenv("ENV_OR_PROD", "DEV")
DEBUG_STATUS = True if ENV_OR_PROD == "DEV" else False

from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from fastapi_globals import g, GlobalsMiddleware

from routers import whaleanalysis, param
from dependencies import Tags

from auth import get_api_key
from lib import timeit

import database as db

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
DESCRIPTION = """
## Backend of Quantist.io: Capturing The Silhouette of Data
_Democratize the information of Indonesia Stocks and more to investors from retail to institution._

Consists of high-end analysis tools based on data with top-down analysis:
* Conservation of money (forex/crypto - bond - stocks)
* External factor (commodities, national/international issues, sentiments)
* Fundamental: business performance, corporate action
* Transaction: foreign flow, broker summary clustering, done trade, 5% shares holder, share holder distribution
* Behaviour: supply & demand, momentum, trend, time
"""

# Get Default Param
@asynccontextmanager
async def lifespan(app: FastAPI):
	# Load default_param
	DEFAULT_PARAM = await db.get_default_param()
	LIST_STOCK = await db.get_list_stock()
	g.set_default("DEFAULT_PARAM", DEFAULT_PARAM)
	g.set_default("LIST_STOCK", LIST_STOCK)
	yield
	# Release default_param
	del DEFAULT_PARAM

# INITIATE APP
app = FastAPI(
	default_response_class=ORJSONResponse,
	debug=DEBUG_STATUS,
	dependencies=[Depends(get_api_key)],
	lifespan=lifespan,
	title="quantist_api",
	description=DESCRIPTION,
	version="0.0.0",
	contact={
		"name": "Kevin Daffa Arrahman",
		"url": "https://quantist.io",
		"email": "kevindaffaarr@quantist.io"
	},
	openapi_tags=[dict(tag.value) for tag in Tags]
)

# CORS https://fastapi.tiangolo.com/tutorial/cors/
ALLOW_ORIGINS:list = os.getenv("ALLOW_ORIGINS","*").split(",")
app.add_middleware(
	CORSMiddleware,
	allow_origins=ALLOW_ORIGINS,
	allow_credentials=True,
	allow_methods=["GET"],
	allow_headers=["*"]
)
app.add_middleware(GlobalsMiddleware)

# INCLUDE ROUTER
app.include_router(whaleanalysis.router)
app.include_router(param.router)

@app.get("/")
@timeit
async def home():
	return {"message": "Welcome to Quantist.io"}
