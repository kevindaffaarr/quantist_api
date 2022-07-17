from fastapi import APIRouter, status, Depends
import datetime
import dependencies as dp
import database as db

# ==========
# Router Initiation
# ==========
router = APIRouter(
	prefix="/whalechart",
	tags=["chart"],
	responses={404: {"description": status.HTTP_404_NOT_FOUND}}
)

# ==========
# Function
# ==========


# ==========
# Router
# ==========
@router.get("/")
def get_whalechart(
	stockcode: str | None = None, startdate: datetime.date | None = None, enddate: datetime.date | None = None,
	dbs: db.Session = Depends(db.get_dbs)
):
	# TODO: (on-going) Quantist Library
	# TODO: stockcode validation based on enum of ListStock + Composite
	pass