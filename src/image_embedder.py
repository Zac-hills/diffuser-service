from transformers import AutoImageProcessor, FlaxResNetModel, ResNetConfig
import torch
from PIL import Image

config = ResNetConfig(out_features=["stage2"])
processor = AutoImageProcessor.from_pretrained(
    "microsoft/resnet-50")
model = FlaxResNetModel.from_pretrained(
    "microsoft/resnet-50", config=config, ignore_mismatched_sizes=True)


def embed_image(image: Image):
    inputs = processor(image, return_tensors="np")
    print(model)
    with torch.no_grad():
        embeddings = model(**inputs)
    print(embeddings)
    return embeddings
