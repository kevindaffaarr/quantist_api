from functools import wraps
from time import perf_counter

import asyncio
import nest_asyncio
nest_asyncio.apply()

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

"""
Helper Function to resolve async function with asyncio with already running event loop
"""
def resolve_async(func_with_args):
	loop = asyncio.get_event_loop()
	return loop.run_until_complete(func_with_args)