from pydantic import BaseModel


class Mask(BaseModel):
    x: int
    y: int
    width: int
    height: int
