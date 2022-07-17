from fastapi import APIRouter, status, Depends
import dp
import db

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
def get_list_code(dbs: db.Session, list_category: dp.ListCategory = "stock", extended:bool = False):
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
@router.get("/list/{list_category}", response_model=list[dp.ListCode], response_model_exclude_unset=True)
def get_list(list_category: dp.ListCategory, extended: bool = False, dbs: db.Session = Depends(db.get_dbs)):
	return get_list_code(dbs=dbs, list_category=list_category, extended=extended)