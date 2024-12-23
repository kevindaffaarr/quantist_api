import os
import warnings
from dotenv import load_dotenv

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi_globals import g, GlobalsMiddleware

from routers import whaleanalysis, param, web
from dependencies import Tags

from auth import get_api_key
from lib import timeit

import database as db

load_dotenv()

# Ignore FutureWarning, DeprecationWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

ENV_PROD_DEV = os.getenv("ENV_PROD_DEV", "DEV")
DEBUG_STATUS = True if ENV_PROD_DEV == "DEV" else False

"""
=============================
FAST API APP
Start the server by run this code in terminal
uvicorn main:app --reload
fastapi run --workers 8 main.py --port 8000
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
	try:
		yield
	except Exception:
		raise
	finally:
		# Release default_param
		del DEFAULT_PARAM

# INITIATE APP
app = FastAPI(
	default_response_class=ORJSONResponse,
	debug=DEBUG_STATUS,
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

app.mount("/static", StaticFiles(directory="pages/static"), name="static")

# INCLUDE ROUTER
app.include_router(whaleanalysis.router, dependencies=[Depends(get_api_key)])
app.include_router(param.router, dependencies=[Depends(get_api_key)])
app.include_router(web.router)

@app.get("")
@app.get("/")
@timeit
async def home():
	return RedirectResponse(url="/web")

if __name__ == "__main__":
	uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)