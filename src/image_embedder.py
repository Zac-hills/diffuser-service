from transformers import AutoImageProcessor,  ViTHybridModel
import torch
from PIL import Image
import numpy as np

processor = AutoImageProcessor.from_pretrained(
    "google/vit-hybrid-base-bit-384")
model = ViTHybridModel.from_pretrained(
    "google/vit-hybrid-base-bit-384")


def embed_image(image: Image) -> np.array:
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**inputs)
    return embeddings.pooler_output.squeeze().cpu().detach().numpy().tolist()
