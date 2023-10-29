from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.preprocessing import MinMaxScaler

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
		self.nbins = await self.calc_nbins() if nbins is None else nbins
		self.size = (self.data['close'].max()-self.data['close'].min())/self.nbins
		self.bins_range = pd.Series(np.arange(self.data['close'].min()-self.size,self.data['close'].max()+self.size,self.size))
		self.hist_bar = self.data.groupby(pd.cut(self.data['close'].to_numpy(),bins=self.bins_range))['netval'].sum() # type:ignore
		self.bins_mid = self.bins_range + self.size/2
		self.peaks_index = await self.calc_peaks_index(self.hist_bar)
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
		if bin_width == 0:
			return 1
		# Calculate the number of nbins
		data_range = self.data["close"].max() - self.data["close"].min()
		nbins = int(data_range/bin_width)
		return nbins
	
	async def calc_peaks_index(self, hist_bar:pd.Series) -> list:
		"""
		Find the maxima and minima of the histogram
		returns list of maxima and minima
		"""
		hist_bar = hist_bar.reset_index(drop=True)
		# Split scaling for positive and negative values
		hist_bar_pos = hist_bar[hist_bar>=0].to_numpy().reshape(-1,1)
		hist_bar_neg = hist_bar[hist_bar<0].to_numpy().reshape(-1,1)
		
		peaks = []
		valleys = []
		scaler = MinMaxScaler()
		if len(hist_bar_pos)>0:
			# Scale the data
			hist_bar_pos = scaler.fit_transform(hist_bar_pos)
			# Find peaks and valleys
			peaks, _ = find_peaks(x=hist_bar_pos.flatten(), prominence=0.1)
			# if peaks or valleys is empty, return max value from hist_bar_pos or hist_bar_neg
			if len(peaks) == 0:
				peaks = [np.argmax(hist_bar_pos).item()]

		if len(hist_bar_neg)>0:
			hist_bar_neg = scaler.fit_transform(-hist_bar_neg)
			valleys, _ = find_peaks(x=hist_bar_neg.flatten(), prominence=0.1)
			if len(valleys) == 0:
				valleys = [np.argmax(hist_bar_neg).item()]
		
		# Find the index of peaks and valleys from the original data
		peaks_ori = hist_bar[hist_bar>=0].index[peaks].to_list()
		valleys_ori = hist_bar[hist_bar<0].index[valleys].to_list()

		# Return the sorted peaks and valleys
		return sorted(peaks_ori + valleys_ori)