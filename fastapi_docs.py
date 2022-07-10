# Contents
# [1] Path Operation, Path Parameter, Query Parameter, Request Body
# [2] Parameters Validation
# [3] Body Parameter and Validation
# [4] Response Model
# [Full Example]

"""
==========
[1] Path Operation, Path Parameter, Query Parameter, Request Body
==========

[Path Operation]
    @app.post("/") #Create Data
    @app.get("/") #Read Data
    @app.put("/") #Update Data
    @app.delete("/") #Delete Data

[Path Parameter (item_id)]
    @app.get("/items/{item_id}")

[Query Parameter (group)]
    @app.get("/items/{item_id}") #Read Data
    async def get_item(item_id: int, group: str)

[Request Body]
    from pydantic import BaseModel
    class Item(BaseModel):
        name:str
        description: str | None = None
        price: float
    @app.post("/items/")
    async def create_item(item: Item):
        return item

    Other Body's important properties look at 3rd part

[Typing]
    https://fastapi.tiangolo.com/tutorial/extra-data-types/
    Typing can be included type, default, and optional
    Python Data Types Annotation:
        common: int, float, Decimal, bool, str, bytes, None, Enum, list, list[str], list[int]
        datetime: datetime.datetime, datetime.date, datetime.time, datetime.timedelta
        other: UUID
    example: group:str | None = None
    meaning: the group parameter is string but it is optional (default value = None).
"""

"""
==========
[2] Parameters Validation
==========
    from fastapi import Path, Query

    # Optional
    q: str = Query(default="this is default value", max_length=16)

    # Mandatory
    q: str = Query(max_length=16)
    q: str = Query(default=..., max_length=16)

    # Mandatory but can be None
    q: str | None = Query(max_length=16)
    q: str | None = Query(default=..., max_length=16)

    # List of properties:
    metadata: title, alias, descripion, deprecated
    string validations: min_length, max_length, regex
    numeric validatons: ge, gt, le, lt
"""

"""
==========
[3] Body Parameter and Validation
==========

    from pydantic import Body, Field

    class Item(BaseModel):
        name: str
        description: str | None = Field(
            default=None, title="The description of the item", max_length=300
        )
        price: float = Field(gt=0, description="The price must be greater than zero")
        tax: float | None = None

    async def update_item(item_id: int, item: Item, user: User, importance: int = Body())
    Nested JSON:
    {
        "item": {
            "name": "Foo",
            "description": "The pretender",
            "price": 42.0,
            "tax": 3.2
        },
        "user": {
            "username": "dave",
            "full_name": "Dave Grohl"
        },
        "importance": 5
    }

    async def update_item(item_id: int, item: Item = Body(embed=True)):
    {
        "item": {
            "name": "Foo",
            "description": "The pretender",
            "price": 42.0,
            "tax": 3.2
        }
    }
"""

"""
==========
[4] Response Model
==========
    response_model, response_model_exclude_unset, response_model_include, response_model_exclude

    meaning:
    response_model_exclude_unset: (return only the values explicitly set)
    response_model_include: (omitting the rest)
    response_model_exclude: (including the rest)

    example
    @app.get("/items/{item_id}", response_model=Item, response_model_exclude_unset=True)

    https://fastapi.tiangolo.com/tutorial/response-model/
"""

# ==========
# [Full Example]
# ==========
from fastapi import FastAPI, Path, Query, status, HTTPException
from pydantic import BaseModel

app = FastAPI()

items = {"foo", "bar"}

class Item(BaseModel):
    name:str
    price: float
    description: str | None = None

@app.post("/items", status_code=status.HTTP_201_CREATED)
async def create_item(item: Item):
    return item

@app.get("/items/{item_id",status_code=status.HTTP_200_OK)
async def get_items(item_id:int, group:str | None = None, availability:bool = True):
    if item_id not in items:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Item not found")
    return {"item_id" : item_id, "group" : group}