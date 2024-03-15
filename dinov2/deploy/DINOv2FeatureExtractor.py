import torch.nn as nn
import torch
from datasets import load_dataset
from transformers import AutoImageProcessor, Dinov2Model


dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = Dinov2Model.from_pretrained("facebook/dinov2-base")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
list(last_hidden_states.shape)


class DINOv2FeatureExtractor(nn.Module):
    """DINOv2 Feature Extractor

    === Class Attributes ===
    - self.ckpt_path: str
        Path to the checkpoint file.
    - self.extraction_model: Dinov2Model
        The DINOv2 model for feature extraction.
    """

    def __init__(self, ckpt_path: str):
        self.ckpt_path = ckpt_path
        image_processor = AutoImageProcessor.from_pretrained(self.ckpt_path)
        self.extraction_model = Dinov2Model.from_pretrained(self.ckpt_path)

    def forward(self, x: torch.Tensor):
        inputs = image_processor(x, return_tensors="pt")
        with torch.no_grad():
            outputs = self.extraction_model(**inputs)
        return outputs.last_hidden_state
