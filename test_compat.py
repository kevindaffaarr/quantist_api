"""
Regression guards for the pandas 3 / pyarrow upgrade.

Both failures below took down the chart and radar endpoints, and neither is
visible until real Decimal/date data flows through, so pin them here with
synthetic frames that need no database.
"""
import datetime
from decimal import Decimal

import pandas as pd
import polars as pl
import pytest

from quantist_library.helper import date_only, pl_to_pandas


def _sample() -> pl.DataFrame:
	# Mirrors what read_database() hands back: SQL NUMERIC -> Decimal(38, 0), DATE -> Date.
	return pl.DataFrame(
		{
			"date": [datetime.date(2024, 1, 1), datetime.date(2024, 1, 2), datetime.date(2024, 1, 3)],
			"tradebleshares": [Decimal(10), Decimal(20), Decimal(30)],
			"listedshares": [Decimal(100), Decimal(100), Decimal(100)],
		},
		schema={"date": pl.Date, "tradebleshares": pl.Decimal(38, 0), "listedshares": pl.Decimal(38, 0)},
	)


def test_pl_to_pandas_gives_native_float_columns():
	df = pl_to_pandas(_sample())
	assert df["tradebleshares"].dtype == "float64"
	assert df["listedshares"].dtype == "float64"


def test_decimal_columns_survive_divide_and_cumsum():
	# Arrow decimal128 raised "Decimal precision out of range [1, 38]: 77" on divide
	# and rejected cumsum outright.
	df = pl_to_pandas(_sample())
	ratio = df["tradebleshares"] / df["listedshares"]
	assert ratio.tolist() == pytest.approx([0.1, 0.2, 0.3])
	assert df["tradebleshares"].cumsum().tolist() == pytest.approx([10.0, 30.0, 60.0])


def test_date_column_compares_against_timestamp():
	df = pl_to_pandas(_sample()).set_index("date")
	assert isinstance(df.index[-1], pd.Timestamp)
	# The radar path filters the date index with pd.to_datetime(startdate).
	assert (df.index >= pd.to_datetime(datetime.date(2024, 1, 2))).sum() == 2


def test_bigquery_datetime_values_are_normalized_before_date_filters():
	values = pl_to_pandas(_sample()).set_index("date").index.tolist()
	assert values
	assert all(type(date_only(value)) is datetime.date for value in values)
	assert date_only(datetime.datetime(2025, 9, 30, 0, 0)) == datetime.date(2025, 9, 30)


def test_last_row_needs_iloc_not_negative_label():
	# pandas 3 treats Series[-1] as a label lookup, so the chart/header code
	# must use .iloc[-1] to read the most recent value.
	s = pl_to_pandas(_sample()).set_index("date")["tradebleshares"]
	assert s.iloc[-1] == 30.0
	with pytest.raises(KeyError):
		s[-1]


def test_groupby_cumsum_and_corrwith_take_no_axis_arg():
	# pandas 3 removed the axis argument from both.
	df = pd.DataFrame({"code": ["A", "A", "B", "B"], "v": [1.0, 2.0, 3.0, 4.0]})
	assert df.groupby("code")["v"].cumsum().tolist() == [1.0, 3.0, 3.0, 7.0]
	other = pd.Series([1.0, 2.0, 3.0, 4.0])
	corr = df.groupby("code")[["v"]].corrwith(other)
	assert list(corr.index) == ["A", "B"]
