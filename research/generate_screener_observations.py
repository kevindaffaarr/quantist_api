"""Collect historical screener observations and publish one static research page.

This is an offline, trigger-driven job. It does not expose an API and it never
uses today's screener result as a historical observation. Run it with explicit
start/end dates after the database contains the required history.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import datetime as dt
import html
import json
import statistics
from pathlib import Path
from typing import Any, Iterable
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import database as db
import dependencies as dp
from routers.web_api import _screener_object

HORIZONS = (1, 3, 5, 10, 20)
METHODS = ("broker", "foreign")
SCREENERS = tuple(entry.value for entry in dp.ScreenerList)
DISCLAIMER = "Bukan sinyal beli/jual atau rekomendasi investasi."
SCHEMA_VERSION = "1.0"


def finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def forward_result(
    observation_date: dt.date,
    sessions: Iterable[tuple[dt.date, float | None, float | None]],
) -> dict[str, Any]:
    """Use only sessions strictly after T; missing closes stay null."""
    future = sorted((row for row in sessions if row[0] > observation_date), key=lambda row: row[0])
    next_open_row = next((row for row in future if finite_number(row[1]) is not None), None)
    next_open = finite_number(next_open_row[1]) if next_open_row else None
    result: dict[str, Any] = {
        "next_session_date": next_open_row[0].isoformat() if next_open_row else None,
        "next_session_open": next_open,
        "horizons": {},
    }
    for horizon in HORIZONS:
        session = future[horizon - 1] if len(future) >= horizon else None
        close = finite_number(session[2]) if session else None
        result["horizons"][str(horizon)] = {
            "future_date": session[0].isoformat() if session else None,
            "future_close": close,
            "forward_return": None if close is None or next_open is None or next_open == 0 else close / next_open - 1,
        }
    return result


def summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for horizon in HORIZONS:
        values = [row["horizons"][str(horizon)]["forward_return"] for row in rows]
        values = [value for value in values if value is not None]
        positive = sum(value > 0 for value in values)
        output[str(horizon)] = {
            "observations": len(rows),
            "valid_results": len(values),
            "positive_count": positive,
            "positive_rate": positive / len(values) if values else None,
            "average": sum(values) / len(values) if values else None,
            "median": statistics.median(values) if values else None,
            "best": max(values) if values else None,
            "worst": min(values) if values else None,
        }
    return output


def date_range_from_database(session: Any, start: dt.date, end: dt.date) -> list[dt.date]:
    values = (
        session.query(db.StockData.date)
        .filter(db.StockData.date.between(start, end))
        .distinct()
        .order_by(db.StockData.date.asc())
        .all()
    )
    return [value[0] for value in values]


def load_prices(session: Any, codes: set[str], start: dt.date, end: dt.date) -> dict[str, list[tuple[dt.date, float | None, float | None]]]:
    if not codes:
        return {}
    values = (
        session.query(db.StockData.code, db.StockData.date, db.StockData.openprice, db.StockData.close)
        .filter(db.StockData.code.in_(sorted(codes)))
        .filter(db.StockData.date.between(start, end))
        .order_by(db.StockData.code.asc(), db.StockData.date.asc())
        .all()
    )
    prices: dict[str, list[tuple[dt.date, float | None, float | None]]] = {}
    for code, date, opening, close in values:
        prices.setdefault(str(code).lower(), []).append((date, finite_number(opening), finite_number(close)))
    return prices


async def collect(args: argparse.Namespace) -> dict[str, Any]:
    start = dt.date.fromisoformat(args.startdate)
    end = dt.date.fromisoformat(args.enddate)
    if start > end:
        raise ValueError("startdate must be on or before enddate")
    slugs = (args.slug,) if args.slug else SCREENERS
    methods = (args.method,) if args.method else METHODS
    session = db.SessionLocal()
    errors: list[str] = []
    observations: list[dict[str, Any]] = []
    try:
        dates = date_range_from_database(session, start, end)
        if args.max_observations:
            dates = dates[: args.max_observations]
        for observation_date in dates:
            for slug in slugs:
                for method_name in methods:
                    try:
                        method = dp.AnalysisMethod(method_name)
                        screener = _screener_object(slug, method, None, observation_date)
                        result = await screener.screen()
                        for rank, (code, values) in enumerate(result.top_stockcodes.iterrows(), start=1):
                            observations.append({
                                "observation_date": observation_date.isoformat(),
                                "slug": slug,
                                "method": method_name,
                                "code": str(code).lower(),
                                "display_name": str(code).upper(),
                                "rank": rank,
                                "reference_close": finite_number(values.get("close")),
                            })
                    except Exception as exc:  # keep other historical runs publishable, but report every failure
                        errors.append(f"{observation_date.isoformat()} {slug} {method_name}: {type(exc).__name__}: {exc}")
        codes = {row["code"] for row in observations}
        prices = load_prices(session, codes, start, end + dt.timedelta(days=args.future_days))
    finally:
        session.close()

    for row in observations:
        future = forward_result(dt.date.fromisoformat(row["observation_date"]), prices.get(row["code"], []))
        row.update(future)
        if row["reference_close"] is None:
            same_day = next((item for item in prices.get(row["code"], []) if item[0].isoformat() == row["observation_date"]), None)
            row["reference_close"] = finite_number(same_day[2]) if same_day else None

    observations.sort(key=lambda row: (row["observation_date"], row["slug"], row["method"], row["rank"], row["code"]))
    if not dates:
        status = "unavailable"
        reason = "Tidak ada tanggal perdagangan pada rentang yang dipilih."
    elif not observations and errors:
        status = "unavailable"
        reason = "Data screener historis tidak dapat dikumpulkan; lihat errors pada artifact."
    elif not observations:
        status = "empty"
        reason = "Tidak ada kandidat screener pada rentang yang dipilih."
    else:
        status = "available"
        reason = None
    dates_with_rows = [dt.date.fromisoformat(row["observation_date"]) for row in observations]
    return {
        "schema_version": SCHEMA_VERSION,
        "study_type": "screener_observation",
        "status": status,
        "disclaimer": DISCLAIMER,
        "reason": reason,
        "query": {"slug": args.slug or "all", "method": args.method or "all", "startdate": args.startdate, "enddate": args.enddate},
        "coverage": {
            "observation_count": len(observations),
            "unique_instruments": len({row["code"] for row in observations}),
            "first_observation": min(dates_with_rows).isoformat() if dates_with_rows else None,
            "last_observation": max(dates_with_rows).isoformat() if dates_with_rows else None,
            "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        },
        "horizons": summarize(observations),
        "observations": observations,
        "errors": errors,
    }


def percent(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.2f}%"


def money_number(value: float | None) -> str:
    return "—" if value is None else f"{value:,.2f}"


def render_html(payload: dict[str, Any]) -> str:
    query = payload["query"]
    coverage = payload["coverage"]
    cards = "".join(
        f'<article><small>{horizon} sesi</small><strong>{percent(payload["horizons"][str(horizon)]["average"])}</strong>'
        f'<span>{payload["horizons"][str(horizon)]["valid_results"]} hasil valid · positif {percent(payload["horizons"][str(horizon)]["positive_rate"])}</span></article>'
        for horizon in HORIZONS
    )
    rows = "".join(
        "<tr>"
        f"<th>{html.escape(row['observation_date'])}</th><td>{html.escape(row['code'].upper())}</td>"
        f"<td>{row['rank']}</td>"
        + "".join(f"<td>{percent(row['horizons'][str(horizon)]['forward_return'])}</td>" for horizon in HORIZONS)
        + "</tr>"
        for row in payload["observations"]
    )
    empty = "" if rows else f'<section class="state"><h2>{"Data belum tersedia" if payload["status"] == "unavailable" else "Tidak ada observasi"}</h2><p>{html.escape(payload["reason"] or "Belum ada hasil pada rentang ini.")}</p></section>'
    error_note = "" if not payload["errors"] else f'<details><summary>{len(payload["errors"])} kesalahan pengumpulan</summary><pre>{html.escape(chr(10).join(payload["errors"]))}</pre></details>'
    table = "" if not rows else f'<section><h2>Daftar observasi</h2><p>Perubahan dihitung dari harga pembukaan sesi berikutnya ke harga penutupan pada horizon terkait.</p><div class="scroll"><table><thead><tr><th>Tanggal observasi</th><th>Ticker</th><th>Peringkat</th>{"".join(f"<th>{horizon} sesi</th>" for horizon in HORIZONS)}</tr></thead><tbody>{rows}</tbody></table></div></section>'
    return f"""<!doctype html>
<html lang="id"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Analisis Observasi Screener · Quantist</title>
<style>
:root{{color-scheme:dark;--page:#0b0c10;--panel:#0e0f14;--raised:#14161d;--border:#232734;--ink:#e6e9f2;--muted:#98a0b4;--purple:#a46bf2}}*{{box-sizing:border-box}}body{{margin:0;background:var(--page);color:var(--ink);font:14px/1.55 Inter,system-ui,sans-serif}}main{{max-width:1200px;margin:auto;padding:48px 28px 72px}}header{{display:flex;justify-content:space-between;gap:32px;border-bottom:1px solid var(--border);padding-bottom:28px}}h1{{font-size:32px;line-height:1.1;margin:6px 0 12px}}h2{{font-size:18px;margin:0 0 8px}}p{{color:var(--muted);margin:6px 0}}.kicker,small{{color:var(--purple);font-size:11px;letter-spacing:.1em;text-transform:uppercase}}.disclaimer{{color:var(--ink);font-weight:600}}.source{{text-align:right;font-size:12px}}code,pre,td,th,strong{{font-variant-numeric:tabular-nums;font-family:ui-monospace,SFMono-Regular,monospace}}section{{margin-top:32px}}.cards{{display:grid;grid-template-columns:repeat(5,1fr);gap:1px;background:var(--border);border:1px solid var(--border)}}article{{background:var(--raised);padding:18px}}article strong{{display:block;font-size:24px;margin:8px 0}}article span{{display:block;color:var(--muted);font-size:12px}}.state{{border:1px solid var(--border);background:var(--raised);padding:24px}}table{{width:100%;border-collapse:collapse;background:var(--panel)}}th,td{{text-align:left;border-bottom:1px solid var(--border);padding:10px 12px;white-space:nowrap}}th{{color:var(--muted);font-size:11px;text-transform:uppercase}}.scroll{{overflow:auto;border:1px solid var(--border)}}details{{margin-top:24px;color:var(--muted)}}@media(max-width:700px){{main{{padding:28px 16px}}header{{display:block}}.source{{text-align:left;margin-top:18px}}.cards{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body><main><header><div><div class="kicker">Riset kuantitatif · hasil statis</div><h1>Analisis Observasi Screener</h1><p class="disclaimer">{html.escape(payload['disclaimer'])}</p><p>Screener adalah penyaring kandidat. Halaman ini mengukur apa yang terjadi setelah observasi tanpa menyebutnya sebagai keputusan transaksi.</p></div><div class="source">Rentang<br><strong>{html.escape(query['startdate'])} – {html.escape(query['enddate'])}</strong><br>{html.escape(str(query['slug']))} · {html.escape(str(query['method']))}</div></header><section><h2>Cakupan</h2><p>{coverage['observation_count']} observasi · {coverage['unique_instruments']} instrumen · dibuat {html.escape(coverage['generated_at'])}</p><div class="cards">{cards}</div></section>{empty}{table}{error_note}</main></body></html>"""


def write_artifacts(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "screener-observations.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ["observation_date", "slug", "method", "code", "display_name", "rank", "reference_close", "next_session_date", "next_session_open"]
    fields += [f"horizon_{horizon}_date" for horizon in HORIZONS] + [f"horizon_{horizon}_close" for horizon in HORIZONS] + [f"horizon_{horizon}_return" for horizon in HORIZONS]
    with (output_dir / "screener-observations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in payload["observations"]:
            flat = {key: row.get(key) for key in fields if key in row}
            for horizon in HORIZONS:
                result = row["horizons"][str(horizon)]
                flat[f"horizon_{horizon}_date"] = result["future_date"]
                flat[f"horizon_{horizon}_close"] = result["future_close"]
                flat[f"horizon_{horizon}_return"] = result["forward_return"]
            writer.writerow(flat)
    (output_dir / "research.html").write_text(render_html(payload), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--startdate", required=True, help="First observation date, YYYY-MM-DD")
    parser.add_argument("--enddate", required=True, help="Last observation date, YYYY-MM-DD")
    parser.add_argument("--slug", choices=SCREENERS, help="One screener; omit to collect all")
    parser.add_argument("--method", choices=METHODS, help="One method; omit to collect broker and foreign")
    parser.add_argument("--future-days", type=int, default=90, help="Calendar-day look-ahead window for future prices")
    parser.add_argument("--max-observations", type=int, default=None, help="Testing/maintenance bound")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[2] / "quantist_web" / "public" / "research")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = asyncio.run(collect(arguments))
    write_artifacts(result, arguments.output_dir)
    print(json.dumps({"status": result["status"], "observations": result["coverage"]["observation_count"], "output_dir": str(arguments.output_dir), "errors": len(result["errors"])}, ensure_ascii=False))
