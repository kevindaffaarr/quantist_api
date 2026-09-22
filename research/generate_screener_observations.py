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

HORIZONS = (1, 3, 5, 10, 20, 50, 100)
METHODS = ("broker", "foreign")
SCREENERS = tuple(entry.value for entry in dp.ScreenerList)
DISCLAIMER = "Bukan sinyal beli/jual atau rekomendasi investasi."
SCHEMA_VERSION = "1.1"


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
    initial_summary = json.dumps(payload["horizons"], ensure_ascii=False)
    initial_reason = html.escape(payload["reason"] or "Belum ada hasil pada rentang ini.")
    error_note = "" if not payload["errors"] else f'<details><summary>{len(payload["errors"])} kesalahan pengumpulan</summary><pre>{html.escape(chr(10).join(payload["errors"]))}</pre></details>'
    return f"""<!doctype html>
<html lang="id"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Analisis Observasi Screener · Quantist</title>
<style>
:root{{--page:#0b0c10;--panel:#0e0f14;--raised:#14161d;--border:#232734;--ink:#e6e9f2;--muted:#98a0b4;--purple:#a46bf2;--green:#62d39a}}*{{box-sizing:border-box}}body{{margin:0;background:var(--page);color:var(--ink);font:14px/1.55 Inter,system-ui,sans-serif}}main{{max-width:1240px;margin:auto;padding:44px 28px 72px}}header{{display:flex;justify-content:space-between;gap:32px;border-bottom:1px solid var(--border);padding-bottom:28px}}h1{{font-size:32px;line-height:1.1;margin:6px 0 12px}}h2{{font-size:18px;margin:0 0 8px}}p{{color:var(--muted);margin:6px 0}}.kicker,small{{color:var(--purple);font-size:11px;letter-spacing:.1em;text-transform:uppercase}}.disclaimer{{color:var(--ink);font-weight:600}}.source{{text-align:right;font-size:12px}}code,pre,td,th,strong,input,select{{font-variant-numeric:tabular-nums;font-family:ui-monospace,SFMono-Regular,monospace}}section{{margin-top:30px}}.filters{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;padding:18px;background:var(--raised);border:1px solid var(--border)}}label{{display:grid;gap:6px;color:var(--muted);font-size:12px}}input,select{{width:100%;min-height:38px;padding:8px 10px;background:var(--panel);color:var(--ink);border:1px solid var(--border);border-radius:6px}}.filter-pair{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}.summary-head{{display:flex;align-items:end;justify-content:space-between;gap:16px}}.summary-head p{{text-align:right}}.scroll{{overflow:auto;border:1px solid var(--border)}}table{{width:100%;border-collapse:collapse;background:var(--panel)}}th,td{{text-align:left;border-bottom:1px solid var(--border);padding:11px 12px;white-space:nowrap}}th{{color:var(--muted);font-size:11px;text-transform:uppercase}}td.number{{text-align:right}}td.positive{{color:var(--green)}}.state{{border:1px solid var(--border);background:var(--raised);padding:24px}}details{{margin-top:24px;color:var(--muted)}}.note{{font-size:12px}}@media(max-width:780px){{main{{padding:28px 16px}}header{{display:block}}.source{{text-align:left;margin-top:18px}}.filters{{grid-template-columns:1fr 1fr}}}}@media(max-width:480px){{.filters{{grid-template-columns:1fr}}.summary-head{{display:block}}.summary-head p{{text-align:left}}}}
</style></head><body><main><header><div><div class="kicker">Riset kuantitatif · hasil statis</div><h1>Analisis Observasi Screener</h1><p class="disclaimer">{html.escape(payload['disclaimer'])}</p><p>Screener adalah penyaring kandidat. Halaman ini mengukur perubahan setelah observasi, bukan keputusan transaksi.</p></div><div class="source">Dataset<br><strong>{html.escape(str(query['slug']))} · {html.escape(str(query['method']))}</strong><br>{html.escape(query['startdate'])} – {html.escape(query['enddate'])}</div></header>
<section><h2>Filter rekap</h2><p>Filter hanya menghitung ulang observasi yang sudah tersimpan di browser. Tidak ada backtest baru saat filter diubah.</p><div class="filters"><label>Rentang tanggal<div class="filter-pair"><input id="date-from" type="date" aria-label="Tanggal mulai"><input id="date-to" type="date" aria-label="Tanggal akhir"></div></label><label>Rentang nilai saham<div class="filter-pair"><input id="value-min" type="number" step="any" placeholder="Minimum" aria-label="Nilai saham minimum"><input id="value-max" type="number" step="any" placeholder="Maksimum" aria-label="Nilai saham maksimum"></div></label><label>Pemilih saham<select id="stock-picker" aria-label="Pemilih saham"><option value="">Semua saham</option></select></label><label>Status data<div id="load-status" class="state">Memuat data tersimpan…</div></label></div></section>
<section><div class="summary-head"><div><h2>Agregat hasil</h2><p>Mean dan median adalah perubahan relatif dari pembukaan sesi berikutnya ke penutupan pada horizon terkait. Hasil positif = persentase observasi dengan perubahan di atas nol.</p></div><p id="row-count">—</p></div><div id="summary" class="scroll"><div class="state">{initial_reason}</div></div></section>{error_note}</main>
<script>
const HORIZONS = [1, 3, 5, 10, 20, 50, 100];
const INITIAL_SUMMARY = {initial_summary};
const state = {{ payload: null }};
const $ = (id) => document.getElementById(id);
const percent = (value) => value == null ? '—' : `${{(value * 100).toFixed(2)}}%`;
const dateValue = (id) => $(id).value;
const finite = (value) => Number.isFinite(Number(value)) ? Number(value) : null;
function median(values) {{
  if (!values.length) return null;
  const ordered = [...values].sort((a, b) => a - b);
  const middle = Math.floor(ordered.length / 2);
  return ordered.length % 2 ? ordered[middle] : (ordered[middle - 1] + ordered[middle]) / 2;
}}
function aggregate(rows) {{
  const groups = new Map();
  for (const row of rows) {{
    const key = `${{row.slug}}|${{row.method}}`;
    if (!groups.has(key)) groups.set(key, {{ slug: row.slug, method: row.method, rows: [] }});
    groups.get(key).rows.push(row);
  }}
  return [...groups.values()].flatMap((group) => HORIZONS.map((horizon) => {{
    const values = group.rows.map((row) => finite(row.horizons?.[String(horizon)]?.forward_return)).filter((value) => value !== null);
    const positive = values.filter((value) => value > 0).length;
    return {{ slug: group.slug, method: group.method, horizon, observations: group.rows.length, valid_results: values.length, mean: values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null, median: median(values), positive_rate: values.length ? positive / values.length : null }};
  }}));
}}
function filteredRows() {{
  const from = dateValue('date-from'); const to = dateValue('date-to');
  const min = finite($('value-min').value); const max = finite($('value-max').value); const stock = $('stock-picker').value;
  return state.payload.observations.filter((row) => {{
    const value = finite(row.reference_close);
    return (!from || row.observation_date >= from) && (!to || row.observation_date <= to) && (min === null || (value !== null && value >= min)) && (max === null || (value !== null && value <= max)) && (!stock || row.code === stock);
  }});
}}
function render() {{
  if (!state.payload) return;
  const rows = filteredRows(); const aggregates = aggregate(rows);
  $('row-count').textContent = `${{rows.length.toLocaleString('id-ID')}} observasi tersaring`;
  $('load-status').textContent = `${{state.payload.coverage.observation_count.toLocaleString('id-ID')}} observasi tersimpan`;
  if (!aggregates.length) {{ $('summary').innerHTML = '<div class="state">Tidak ada observasi yang cocok dengan filter.</div>'; return; }}
  $('summary').innerHTML = `<table><thead><tr><th>Screener</th><th>Metode</th><th>Horizon</th><th>Observasi</th><th>Hasil valid</th><th>Rata-rata</th><th>Median</th><th>Hasil positif</th></tr></thead><tbody>${{aggregates.map((item) => `<tr><td>${{item.slug}}</td><td>${{item.method}}</td><td>${{item.horizon}} sesi</td><td class="number">${{item.observations.toLocaleString('id-ID')}}</td><td class="number">${{item.valid_results.toLocaleString('id-ID')}}</td><td class="number">${{percent(item.mean)}}</td><td class="number">${{percent(item.median)}}</td><td class="number positive">${{percent(item.positive_rate)}}</td></tr>`).join('')}}</tbody></table>`;
}}
function populateStocks() {{
  const stocks = [...new Map(state.payload.observations.map((row) => [row.code, row.display_name || row.code.toUpperCase()])).entries()].sort((a, b) => a[1].localeCompare(b[1]));
  $('stock-picker').innerHTML = '<option value="">Semua saham</option>' + stocks.map(([code, name]) => `<option value="${{code}}">${{name}}</option>`).join('');
}}
async function load() {{
  try {{
    const response = await fetch('screener-observations.json', {{ cache: 'no-store' }});
    if (!response.ok) throw new Error(`HTTP ${{response.status}}`);
    state.payload = await response.json();
    $('date-from').value = state.payload.query.startdate; $('date-to').value = state.payload.query.enddate;
    populateStocks(); render();
  }} catch (error) {{ $('load-status').textContent = `Data tidak dapat dimuat: ${{error.message}}`; $('summary').innerHTML = '<div class="state">Data historis belum tersedia.</div>'; }}
}}
['date-from', 'date-to', 'value-min', 'value-max', 'stock-picker'].forEach((id) => $(id).addEventListener('input', render));
load();
</script></body></html>"""


def write_artifacts(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "screener-observations.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ["observation_date", "slug", "method", "code", "display_name", "rank", "reference_close", "next_session_date", "next_session_open"]
    fields += [f"horizon_{horizon}_date" for horizon in HORIZONS] + [f"horizon_{horizon}_close" for horizon in HORIZONS] + [f"horizon_{horizon}_return" for horizon in HORIZONS]
    with (output_dir / "screener-observations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
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
    parser.add_argument("--future-days", type=int, default=365, help="Calendar-day look-ahead window for future prices")
    parser.add_argument("--max-observations", type=int, default=None, help="Testing/maintenance bound")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[2] / "quantist_web" / "public" / "research")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = asyncio.run(collect(arguments))
    write_artifacts(result, arguments.output_dir)
    print(json.dumps({"status": result["status"], "observations": result["coverage"]["observation_count"], "output_dir": str(arguments.output_dir), "errors": len(result["errors"])}, ensure_ascii=False))
