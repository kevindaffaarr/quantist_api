from functools import wraps
from time import perf_counter
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

# Determine the number of nbins by Freedman-Diaconis rule
async def calc_bins(data:pd.DataFrame) -> int:
    # Calculate IQR from data["close"]
    q1 = (data["close"]).quantile(0.25)
    q3 = (data["close"]).quantile(0.75)
    iqr = q3 - q1
    # State the number of data
    n = len(data["netval"])
    # Calculate the bin width
    bin_width = 2*iqr/(n**(1/3))
    # Calculate the number of nbins
    data_range = data["close"].max() - data["close"].min()
    nbins = int(data_range/bin_width)
    return nbins