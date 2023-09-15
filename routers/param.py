from fastapi import APIRouter, status, Depends
import dependencies as dp
import database as db
from lib import timeit

# ==========
# Router Initiation
# ==========
router = APIRouter(
	prefix="/param",
	tags=["param"],
	responses={404: {"description": status.HTTP_404_NOT_FOUND}}
)

# ==========
# Function
# ==========
async def get_list_code(dbs: db.Session, list_category: dp.ListCategory = dp.ListCategory.stock, extended:bool = False):
	if list_category == "stock" and extended == True:
		return dbs.query(db.ListStock).all()
	elif list_category == "stock":
		return dbs.query(db.ListStock.index, db.ListStock.code).all()
	elif list_category == "broker":
		return dbs.query(db.ListBroker.index, db.ListBroker.code).all()
	elif list_category == "index":
		return dbs.query(db.ListIndex.index, db.ListIndex.code).all()
	else:
		return status.HTTP_404_NOT_FOUND

# ==========
# Router
# ==========
@router.get("/dataparam", response_model=list[dp.DataParam])
@timeit
async def get_dataparam(key:str|None = None, dbs: db.Session = Depends(db.get_dbs)):
	if key is None:
		return dbs.query(db.DataParam).all()
	if key:
		return dbs.query(db.DataParam).filter(db.DataParam.param == key).all() # type: ignore
	
	return None

@router.get("/list/{list_category}", response_model=list[dp.ListCode], response_model_exclude_unset=True)
@timeit
async def get_list(list_category: dp.ListCategory, extended: bool = False, dbs: db.Session = Depends(db.get_dbs)):
	return await get_list_code(dbs=dbs, list_category=list_category, extended=extended)