from fastapi import APIRouter

router = APIRouter(
	prefix="/whalechart",
	tags=["chart"],
	responses={404: {"description": "Not found"}}
)

@router.get("/")
async def get_whalechart():
	return {"message" : "This will be whalechart route"}