import datetime as dt
import json

import pandas as pd

from quantist_library.screener import flow_price_correlation
from research.generate_screener_observations import HORIZONS, forward_result, render_html, summarize


def test_forward_result_ignores_prices_on_or_before_observation():
    result = forward_result(
        dt.date(2026, 9, 18),
        [
            (dt.date(2026, 9, 17), 99.0, 100.0),
            (dt.date(2026, 9, 18), 110.0, 111.0),
            (dt.date(2026, 9, 21), 120.0, 121.0),
        ],
    )
    assert result["next_session_date"] == "2026-09-21"
    assert result["next_session_open"] == 120.0
    assert result["horizons"]["1"]["forward_return"] == 121 / 120 - 1


def test_forward_result_does_not_fill_missing_future_close():
    result = forward_result(
        dt.date(2026, 9, 18),
        [
            (dt.date(2026, 9, 21), 120.0, 121.0),
            (dt.date(2026, 9, 22), 130.0, None),
            (dt.date(2026, 9, 23), 140.0, 143.0),
        ],
    )
    assert result["horizons"]["3"]["future_date"] == "2026-09-23"
    assert result["horizons"]["5"]["future_close"] is None
    assert result["horizons"]["5"]["forward_return"] is None


def test_zero_next_open_is_unavailable_not_an_exception():
    result = forward_result(
        dt.date(2026, 9, 18),
        [(dt.date(2026, 9, 21), 0.0, 121.0)],
    )
    assert result["horizons"]["1"]["forward_return"] is None


def test_summary_uses_only_valid_numeric_returns():
    rows = [
        {"horizons": {str(horizon): {"forward_return": 0.1 if horizon == 1 else None} for horizon in HORIZONS}},
        {"horizons": {str(horizon): {"forward_return": -0.1 if horizon == 1 else None} for horizon in HORIZONS}},
    ]
    summary = summarize(rows)["1"]
    assert summary["observations"] == 2
    assert summary["valid_results"] == 2
    assert summary["positive_count"] == 1
    assert summary["positive_rate"] == 0.5
    assert summary["average"] == 0.0
    assert summary["median"] == 0.0


def test_summary_includes_50_and_100_session_horizons():
    rows = [{"horizons": {str(horizon): {"forward_return": 0.05} for horizon in HORIZONS}}]
    summary = summarize(rows)
    assert tuple(int(horizon) for horizon in summary) == (1, 3, 5, 10, 20, 50, 100)
    assert summary["50"]["median"] == 0.05
    assert summary["100"]["positive_rate"] == 1.0


def test_single_code_flow_correlation_has_a_flat_index():
    index = pd.MultiIndex.from_product([["bbri"], pd.date_range("2026-06-10", periods=3)], names=["code", "date"])
    close = pd.Series([100.0, 101.0, 103.0], index=index)
    net_value = pd.Series([10.0, 15.0, 20.0], index=index)
    result = flow_price_correlation(close, net_value)
    assert list(result.index) == ["bbri"]
    assert result.index.nlevels == 1


def test_static_html_contains_indonesian_disclaimer_and_no_trade_language():
    payload = {
        "query": {"slug": "vwap_rally", "method": "foreign", "startdate": "2026-01-01", "enddate": "2026-09-18"},
        "coverage": {"observation_count": 0, "unique_instruments": 0, "generated_at": "2026-09-22T00:00:00+00:00"},
        "disclaimer": "Bukan sinyal beli/jual atau rekomendasi investasi.",
        "status": "unavailable",
        "reason": "Data historis belum tersedia.",
        "horizons": {str(horizon): {"average": None, "valid_results": 0, "positive_rate": None} for horizon in HORIZONS},
        "observations": [],
        "errors": [],
    }
    html = render_html(payload)
    assert "Analisis Observasi Screener" in html
    assert payload["disclaimer"] in html
    assert "Data historis belum tersedia." in html
    assert "BUY" not in html and "SELL" not in html
    assert "Rentang tanggal" in html
    assert "Rentang nilai saham" in html
    assert "Pemilih saham" in html
    assert "Rata-rata" in html and "Median" in html and "Hasil positif" in html
    json.dumps(payload)

