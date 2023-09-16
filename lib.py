from __future__ import annotations
from functools import wraps
from time import perf_counter
import numpy as np
import pandas as pd

# import asyncio
# import nest_asyncio
# nest_asyncio.apply()

"""
Function for decorator to calculate the time taken by a function
"""
def timeit(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = perf_counter()
        result = await func(*args, **kwargs)
        end = perf_counter()
        print(f"{func.__name__} took {end - start} seconds")
        return result
    return wrapper

# """
# Helper Function to resolve async function with asyncio with already running event loop
# """
# def resolve_async(func_with_args):
# 	loop = asyncio.get_event_loop()
# 	return loop.run_until_complete(func_with_args)

class Bin():
    def __init__(self, data:pd.DataFrame) -> None:
        self.data:pd.DataFrame = data
        self.nbins:int
        self.size:float
        self.bins_range:pd.Series
        self.hist_bar:pd.Series
        self.bins_mid:pd.Series

    async def fit(self, nbins:int | None = None) -> Bin:
        self.nbins = await self.calc_bins() if nbins is None else nbins
        self.size = (self.data['close'].max()-self.data['close'].min())/self.nbins
        self.bins_range = pd.Series(np.arange(self.data['close'].min(),self.data['close'].max(),self.size))
        self.hist_bar = self.data.groupby(pd.cut(self.data['close'].to_numpy(),bins=self.bins_range))['netval'].sum() # type:ignore
        self.bins_mid = self.bins_range + self.size/2
        return self
    
    async def calc_bins(self) -> int:
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