from fastapi import APIRouter, status, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os
from lib import timeit

# Format newRequest format for Jinja2
def format_newRequest(request: Request) -> Request:
	request.scope['scheme'] = os.getenv("API_SCHEME", "https")
	header_list = list(request.scope['headers'])
	# Search for key b"host" in header_list, then erase the b"host" key
	for i, header in enumerate(header_list):
		if header[0] == b"host":
			del header_list[i]
			header_list.append((b"host", os.getenv("API_HOST", "127.0.01").encode('ascii')))
	request.scope['headers'] = tuple(header_list)

	return Request(request.scope)

# ==========
# Router Initiation
# ==========
router = APIRouter(
	prefix="/web",
	tags=["web"],
	responses={404: {"description": status.HTTP_404_NOT_FOUND}},
	dependencies=[Depends(format_newRequest)]
)

def getenv(default:str, key: str) -> str:
	return os.getenv(key, default)

# Template Initiation
templates = Jinja2Templates(directory="pages", autoescape= True, auto_reload= True)
templates.env.filters['getenv'] = getenv

# ==========
# DEFAULT ROUTER
# ==========
@router.get("", response_class=HTMLResponse, status_code=status.HTTP_200_OK)
@router.get("/", response_class=HTMLResponse, status_code=status.HTTP_200_OK)
@timeit
async def index(request: Request, name: str|None = None):
	return templates.TemplateResponse(request=request, name="index.html", context={"name": name})

# TODO: router with query parameters
# TODO: screener
