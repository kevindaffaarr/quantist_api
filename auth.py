from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
import os
import json

# Get string of dict of REGISTERED_API_KEYS from environment variables
# Then the string converted to dict by json.loads
# String format: {"api_key1":"secret1","api_key2":"secret2"}
REGISTERED_API_KEYS: str = os.getenv("REGISTERED_API_KEYS")
REGISTERED_API_KEYS: dict = json.loads(REGISTERED_API_KEYS)
API_KEY_NAME = "X-API-KEY"

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Security function by API Key for fastapi
async def get_api_key(api_key: str|None = Security(api_key_header)):
    if api_key not in REGISTERED_API_KEYS.keys():
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    return REGISTERED_API_KEYS[api_key]
