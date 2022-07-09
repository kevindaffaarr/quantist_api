from fastapi import FastAPI
from sqlalchemy import desc
from .routers import whalechart

"""
=============================
FAST API APP
Start the server by run this code in terminal
uvicorn main:app --reload
=============================

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

app = FastAPI(
    title="quantist_api",
    description=description,
    version="0.0.1",
    contact={
        "name": "Kevin Daffa Arrahman",
        "email": "kevindaffaarr@quantist.io"
    }
)

app.include_router(whalechart.router)

@app.get("/")
async def home():
    return {"message": "Welcome to Quantist.io"}