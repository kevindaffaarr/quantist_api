from fastapi import APIRouter, status, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os
from lib import timeit

# ==========
# Router Initiation
# ==========
router = APIRouter(
	prefix="/web",
	tags=["web"],
	responses={404: {"description": status.HTTP_404_NOT_FOUND}}
)

def getenv(default:str, key: str) -> str:
	return os.getenv(key, default)

# Template Initiation
templates = Jinja2Templates(directory="pages", autoescape= True, auto_reload= True)
templates.env.filters['getenv'] = getenv

# ==========
# DEFAULT ROUTER
# ==========
@router.get("/", response_class=HTMLResponse, status_code=status.HTTP_200_OK)
@timeit
async def index(request: Request, name: str|None = None):
    return templates.TemplateResponse(request=request, name="index.html", context={"name": name})