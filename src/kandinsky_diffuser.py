from diffusers import DiffusionPipeline, KandinskyInpaintPipeline, KandinskyPriorPipeline
from util import image_to_byte_array
from fastapi.exceptions import HTTPException
from PIL import Image
import torch
import numpy as np
from mask import Mask

pipe_prior = DiffusionPipeline.from_pretrained(
    "kandinsky-community/kandinsky-2-1-prior", torch_dtype=torch.float16)
pipe_prior.to("cuda")

t2i_pipe = DiffusionPipeline.from_pretrained(
    "kandinsky-community/kandinsky-2-1", torch_dtype=torch.float16)
t2i_pipe.to("cuda")

edit_pipeline_embedder = KandinskyPriorPipeline.from_pretrained(
    "kandinsky-community/kandinsky-2-1-prior", torch_dtype=torch.float16
)
edit_pipeline_embedder.to("cuda")

edit_pipeline = KandinskyInpaintPipeline.from_pretrained(
    "kandinsky-community/kandinsky-2-1-inpaint", torch_dtype=torch.float16)
edit_pipeline.to("cuda")

width_2k = 2560
height_2k = 1440


def text_to_image(prompt: str, width: int = 1920, height: int = 1080) -> bytes:
    width = min(max(width, 0), width_2k)
    height = min(max(height, 0), height_2k)
    negative_prompt = "low quality, bad quality"
    image_embeds, negative_image_embeds = pipe_prior(
        prompt, negative_prompt, guidance_scale=1.0).to_tuple()
    image = t2i_pipe(
        prompt, image_embeds=image_embeds, negative_image_embeds=negative_image_embeds, height=height, width=width
    ).images[0]
    return image_to_byte_array(image)


def edit_image(image: Image, prompt: str, mask_obj: Mask) -> bytes:
    width, height = image.size
    if width % 8 != 0 or height % 8 != 0:
        raise HTTPException(
            409, detail="Error image width and height MUST be a multiple of 8")
    print(image.size)
    mask = np.ones((height, width), dtype=np.float32)
    # apply mask to the area the model should edit
    mask[mask_obj.y:mask_obj.y+mask_obj.height,
         mask_obj.x:mask_obj.x+mask_obj.width] = 0
    prior_output = edit_pipeline_embedder(prompt)
    out = edit_pipeline(
        prompt,
        image=image,
        mask_image=mask,
        **prior_output,
        height=height,
        width=width,
        num_inference_steps=150,
    ).images[0]
    print(out)
    return image_to_byte_array(out)
