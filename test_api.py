import os
import pytest
import requests
from dotenv import load_dotenv
load_dotenv()

# ===
# Run with: pytest -s -v test_api.py
# ===

URL = os.getenv("TESTING_API_URL", "http://127.0.0.1:8000")
HEADERS = {"X-API-KEY": os.getenv("TESTING_API_KEY")}

# Create api list from method inside files in routes folder
PATH = [
    "/param/dataparam",
    "/param/list/stock",
    "/",
    "/whaleanalysis/chart", "/whaleanalysis/chart/foreign", "/whaleanalysis/chart/broker",
    "/whaleanalysis/radar", "/whaleanalysis/radar/foreign", "/whaleanalysis/radar/broker",
    "/whaleanalysis/full-data", "/whaleanalysis/full-data/foreign", "/whaleanalysis/full-data/broker",
    "/whaleanalysis/screener", 
    "/whaleanalysis/screener/foreign",
    "/whaleanalysis/screener/foreign/top-money-flow?accum_or_distri=most_accumulated", "/whaleanalysis/screener/foreign/top-money-flow?accum_or_distri=most_distributed",
    "/whaleanalysis/screener/foreign/vwap?screener_vwap_criteria=vwap_rally","/whaleanalysis/screener/foreign/vwap?screener_vwap_criteria=vwap_around", "/whaleanalysis/screener/foreign/vwap?screener_vwap_criteria=vwap_breakout", "/whaleanalysis/screener/foreign/vwap?screener_vwap_criteria=vwap_breakdown",
    "/whaleanalysis/screener/foreign/vprofile",
    "/whaleanalysis/screener/broker/top-money-flow?accum_or_distri=most_accumulated", "/whaleanalysis/screener/broker/top-money-flow?accum_or_distri=most_distributed",
    "/whaleanalysis/screener/broker/vwap?screener_vwap_criteria=vwap_rally","/whaleanalysis/screener/broker/vwap?screener_vwap_criteria=vwap_around", "/whaleanalysis/screener/broker/vwap?screener_vwap_criteria=vwap_breakout", "/whaleanalysis/screener/broker/vwap?screener_vwap_criteria=vwap_breakdown",
    "/whaleanalysis/screener/broker/vprofile",
]

# Pytest
@pytest.mark.parametrize("api", PATH)
def test_api(api):
    response = requests.get(f"{URL}{api}", headers=HEADERS)
    assert response.status_code == 200, f"Error: {response.text}"
