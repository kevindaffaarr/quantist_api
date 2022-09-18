import os

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from routers import whaleanalysis, param
from dependencies import Tags

from auth import get_api_key
from lib import timeit

ENV_OR_PROD = os.getenv("ENV_OR_PROD", "DEV")
DEBUG_STATUS = True if ENV_OR_PROD == "DEV" else False

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

# INITIATE APP
app = FastAPI(
	debug=DEBUG_STATUS,
	dependencies=[Depends(get_api_key)],
	title="quantist_api",
	description=DESCRIPTION,
	version="0.0.0",
	contact={
		"name": "Kevin Daffa Arrahman",
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

# INCLUDE ROUTER
app.include_router(whaleanalysis.router)
app.include_router(param.router)

@app.get("/")
@timeit
async def home():
	return {"message": "Welcome to Quantist.io"}
