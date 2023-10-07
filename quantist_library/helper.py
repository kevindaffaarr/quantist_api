from __future__ import annotations
import numpy as np
import pandas as pd

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
		
		"""
		self.nbins = await self.calc_nbins() if nbins is None else nbins
		self.size = (self.data['close'].max()-self.data['close'].min())/self.nbins
		self.bins_range = pd.Series(np.arange(self.data['close'].min()-self.size,self.data['close'].max()+self.size,self.size))
		self.hist_bar = self.data.groupby(pd.cut(self.data['close'].to_numpy(),bins=self.bins_range))['netval'].sum() # type:ignore
		self.bins_mid = self.bins_range + self.size/2
		return self
	
	async def calc_nbins(self) -> int:
		# Calculate IQR from data["close"]
		q1 = (self.data["close"]).quantile(0.25)
		q3 = (self.data["close"]).quantile(0.75)
		iqr = q3 - q1
		# State the number of data
		n = len(self.data["netval"])
		# Calculate the bin width
		bin_width = 2*iqr/(n**(1/3))
		# Calculate the number of nbins
		data_range = self.data["close"].max() - self.data["close"].min()
		nbins = int(data_range/bin_width)
		return nbins