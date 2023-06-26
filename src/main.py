from fastapi import FastAPI, Response, UploadFile, File, Depends
from kandinsky_diffuser import text_to_image, edit_image
from mask import Mask
from image_embedder import embed_image
from pydantic import BaseModel
import io
from PIL import Image

app = FastAPI()


class HTTPError(BaseModel):
    detail: str

    class Config:
        schema_extra = {
            "example": {"detail": "HTTPException raised."},
        }


@app.get("/text-to-image", responses={200: {"content": {"image/png": {}}}}, response_class=Response)
def text_to_image_route(prompt: str, width: int = 1280, height: int = 720):
    return Response(content=text_to_image(prompt, width, height), media_type="image/png")


@app.post("/edit-image", responses={
    200: {"content": {"image/png": {}}},
    409: {
        "model": HTTPError,
        "description": "The image requires width and height to be multiple of 8",
    }, }, response_class=Response)
async def edit_image_route(prompt: str, mask: Mask = Depends(), file: UploadFile = File(...)):
    request_object_content = await file.read()
    img = Image.open(io.BytesIO(request_object_content))
    return Response(content=edit_image(image=img, prompt=prompt, mask_obj=mask), media_type="image/png")


@app.post("/embed-image")
async def embed_image_route(file: UploadFile = File(...)):
    request_object_content = await file.read()
    img = Image.open(io.BytesIO(request_object_content))
    return embed_image(img)
