import os
from pathlib import Path
from typing import List, Union, Tuple

import numpy as np
import torch
from PIL import Image
import open_clip


class ClipMultiModal:
    """
    Wrapper around OpenCLIP to produce a shared embedding space for text and images.
    We use the same model (text and vision towers) so text, images, and text from audio
    all land in one space for cross-modal retrieval.
    """

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model.eval().to(self.device)
        self.tokenizer = open_clip.get_tokenizer(model_name)

    @torch.inference_mode()
    def embed_text(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, self.model.text_projection.shape[1]), dtype=np.float32)
        tokens = self.tokenizer(texts)
        tokens = tokens.to(self.device)
        text_features = self.model.encode_text(tokens)
        if normalize:
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.detach().cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def embed_images(
        self, images: List[Union[str, Path, Image.Image]], normalize: bool = True
    ) -> np.ndarray:
        if len(images) == 0:
            return np.zeros((0, self.model.text_projection.shape[1]), dtype=np.float32)
        processed = []
        for item in images:
            if isinstance(item, (str, Path)):
                img = Image.open(item).convert("RGB")
            else:
                img = item.convert("RGB")
            processed.append(self.preprocess(img))
        image_input = torch.stack(processed).to(self.device)
        image_features = self.model.encode_image(image_input)
        if normalize:
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.detach().cpu().numpy().astype(np.float32)


def batch(iterable, n: int):
    """Yield successive n-sized chunks from iterable."""
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]
