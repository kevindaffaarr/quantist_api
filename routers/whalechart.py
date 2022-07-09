from fastapi import APIRouter

router = APIRouter(
    prefix="/whalechart",
    tags="chart",
    dependencies=[],
    responses={404: {"description": "Not found"}}
)

@router.get("/")
async def whale_chart():
    pass