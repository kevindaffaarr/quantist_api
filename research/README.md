# Static screener observation research

`generate_screener_observations.py` is a trigger-driven offline job. It reads the PostgreSQL database configured by `quantist_api/.env`, runs the existing screener implementation at historical market-close dates, reads future OHLC rows, and writes one static publication bundle.

The first publication does not use an API, Worker, browser fetch, or date archive. Re-running the job overwrites the same output files.

## First three-year publication

Use the latest complete observation date that still has 20 future trading sessions. With the current database snapshot that is `2026-08-21`:

```bash
cd /home/kevindaffaarr/quantist_api
.venv/bin/python research/generate_screener_observations.py \
  --startdate 2023-01-01 \
  --enddate 2026-08-21 \
  --slug vwap_rally \
  --method foreign \
  --output-dir /home/kevindaffaarr/quantist_web/public/research
```

Omit `--slug` and `--method` to collect every supported screener and both methods. That is a larger trigger and should be run deliberately.

## Outputs

```text
quantist_web/public/research/
  research.html
  screener-observations.json
  screener-observations.csv
```

The calculation uses the next valid session open as the comparison price and the close on the 1/3/5/10/20th future session. Missing future prices remain null and are excluded from that horizon's aggregates. The output is a static Indonesian page; it never calls an API.
