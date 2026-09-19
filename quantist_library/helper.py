from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import polars as pl
import datetime


def date_only(value: datetime.date | datetime.datetime | pd.Timestamp | None) -> datetime.date | None:
	"""Return a date-only value suitable for SQL DATE parameters.

	BigQuery rejects datetime values such as ``2025-09-30T00:00:00`` when
	binding them to a DATE column. Database drivers may expose DATE columns as
	Python datetimes or pandas timestamps, so normalize both forms here.
	"""
	if value is None:
		return None
	if isinstance(value, pd.Timestamp):
		return datetime.date(value.year, value.month, value.day)
	if isinstance(value, datetime.datetime):
		return value.date()
	if isinstance(value, datetime.date):
		return value
	raise TypeError(f"Expected a date-like value, got {type(value).__name__}")


class Bin():
	def __init__(self, data:pd.DataFrame) -> None:
		self.data:pd.DataFrame = data
		self.nbins:int
		self.size:float
		self.bins_range:pd.Series
		self.hist_bar:pd.Series
		self.bins_mid:pd.Series

	async def fit(self, nbins:int | None = None) -> Bin:
		"""
		Fit the data into bins
		returns self

		Properties:
		- nbins: number of bins: int
		- size: size of each bin: float
		- bins_range: range of bins: pd.Series
		- hist_bar: histogram of each bin: pd.Series
		- bins_mid: mid point of each bin: pd.Series
		- peaks_index: index of peaks and valleys of hist_bar: list
		"""
		data_close:pd.Series = self.data['close'].astype(float)

		self.nbins = max(1, int(await self.calc_nbins() if nbins is None else nbins))
		data_min = float(data_close.min())
		data_max = float(data_close.max())
		data_range = data_max - data_min
		if data_range <= 0 or not np.isfinite(data_range):
			# A constant price still gets one usable interval for charting and
			# profile fallback selection; zero-width bins cannot be cut.
			self.size = max(abs(data_min) * 1e-6, 1e-6)
			self.bins_range = pd.Series([data_min - self.size, data_max + self.size])
		else:
			self.size = data_range / self.nbins
			self.bins_range = pd.Series(np.arange(
				data_min - self.size,
				data_max + self.size,
				self.size,
			))
		self.hist_bar = self.data.groupby(pd.cut(data_close.to_numpy(),bins=self.bins_range))['netval'].sum() # type:ignore
		self.bins_mid = self.bins_range + self.size/2
		self.peaks_index = await self.calc_peaks_index(self.hist_bar)
		return self
	
	async def calc_nbins(self) -> int:
		data_close:pd.Series = self.data["close"].astype('float')
		# Calculate IQR from data["close"]
		q1 = (data_close).quantile(0.25)
		q3 = (data_close).quantile(0.75)
		iqr = q3 - q1
		# State the number of data
		n = len(self.data["netval"])
		# Calculate the bin width
		bin_width = 2*iqr/(n**(1/3))
		if not np.isfinite(bin_width) or bin_width <= 0:
			return 1
		# Calculate the number of nbins
		data_range = data_close.max() - data_close.min()
		nbins = int(data_range/bin_width)
		return max(1, nbins)
	
	async def calc_peaks_index(self, hist_bar:pd.Series) -> list:
		"""
		Find significant positive peaks and negative-flow valleys.

		Each sign is masked onto the original bin axis, so an empty or
		opposite-sign bin remains a separator rather than disappearing from
		the neighbourhood used by ``find_peaks``. A local node is significant
		when its raw prominence clears both 10% of the strongest absolute
		histogram value and twice the median absolute deviation of its masked
		flow. If there are no local extrema, the strongest absolute bin is
		retained as a deterministic fallback.
		"""
		values = pd.to_numeric(hist_bar.reset_index(drop=True), errors="coerce")
		values = values.fillna(0.0).to_numpy(dtype=float, copy=True)
		values[~np.isfinite(values)] = 0.0
		if len(values) == 0:
			return []

		absolute_values = np.abs(values)
		strongest = float(absolute_values.max())
		if strongest <= 0:
			return []
		minimum_prominence = strongest * 0.1
		candidates: list[int] = []
		has_local_extrema = False
		for flow in (
			np.where(values > 0, values, 0.0),
			np.where(values < 0, -values, 0.0),
		):
			flow_median = float(np.median(flow))
			flow_mad = float(np.median(np.abs(flow - flow_median)))
			prominence_threshold = max(minimum_prominence, 2 * flow_mad)
			local, properties = find_peaks(flow, prominence=0)
			if len(local):
				has_local_extrema = True
				prominences = properties["prominences"] if "prominences" in properties else np.zeros(len(local))
				candidates.extend(
					int(index)
					for index, prominence in zip(local, prominences)
					if prominence >= prominence_threshold
				)

		if candidates:
			return sorted(set(candidates))
		if not has_local_extrema:
			return [int(np.argmax(absolute_values))]
		return []

def pl_to_pandas(df: pl.DataFrame) -> pd.DataFrame:
	"""
	Convert a polars frame read from the database into native pandas dtypes.

	SQL NUMERIC columns arrive as Decimal(38, 0). Left as Arrow decimal128 they
	blow past Arrow's 38-digit limit on divide (precision 77) and reject cumsum
	outright, so cast them to float64 before handing pandas numpy-backed columns.
	"""
	return df.cast({pl.Decimal: pl.Float64}).to_pandas(use_pyarrow_extension_array=False)
