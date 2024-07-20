import os
import pytest
import requests
from dotenv import load_dotenv
load_dotenv()

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
    "/whaleanalysis/screener/foreign", "/whaleanalysis/screener/foreign/top-money-flow", "/whaleanalysis/screener/foreign/vwap", "/whaleanalysis/screener/foreign/vprofile",
    "/whaleanalysis/screener/broker/top-money-flow", "/whaleanalysis/screener/broker/vwap", "/whaleanalysis/screener/broker/vprofile"
]

# Pytest
@pytest.mark.parametrize("api", PATH)
def test_api(api):
    response = requests.get(f"{URL}{api}", headers=HEADERS)
    assert response.status_code == 200, f"Error: {response.text}"
