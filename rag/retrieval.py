from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .models import ClipMultiModal
from .indexing import MultiModalIndex


@dataclass
class RetrievalResult:
    score: float
    modality: str
    metadata: Dict[str, Any]


class Retriever:
    """High-level retrieval wrapper to support text and image queries."""

    def __init__(self, index: MultiModalIndex, clip: ClipMultiModal):
        self.index = index
        self.clip = clip

    def search_text(self, query: str, k: int = 6) -> List[RetrievalResult]:
        q = self.clip.embed_text([query])[0].tolist()
        res = self.index.query(q, k=k)
        return self._format_results(res)
    
    def search_cross_modal(self, query: str, k: int = 10, boost_images: bool = False) -> List[RetrievalResult]:
        """Enhanced cross-modal search that finds both text and images for a text query"""
        # Use text embedding to find content in the shared CLIP space
        q = self.clip.embed_text([query])[0].tolist()
        # Get more results initially to have better selection
        res = self.index.query(q, k=k * 2)
        
        results = self._format_results(res)
        
        if boost_images:
            # Apply stronger boosting for image results in cross-modal search
            for result in results:
                if (result.metadata.get('modality') == 'image' or 
                    result.metadata.get('type') == 'image_text'):
                    # Strong boost for images in cross-modal search
                    result.score = min(0.95, result.score * 1.5)
        
        # Sort by score and return top k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]

    def search_image(self, image_path: Path, k: int = 6) -> List[RetrievalResult]:
        q = self.clip.embed_images([image_path])[0].tolist()
        res = self.index.query(q, k=k)
        return self._format_results(res)

    def _format_results(self, res) -> List[RetrievalResult]:
        results: List[RetrievalResult] = []
        ids = res.get("ids", [[]])[0]
        dists = res.get("distances", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        for i, (idv, dv, mv) in enumerate(zip(ids, dists, metas)):
            # Chroma returns distance (cosine distance). Convert to similarity score.
            sim = 1.0 - float(dv) if dv is not None else 0.0
            results.append(
                RetrievalResult(
                    score=sim,
                    modality=mv.get("modality", mv.get("type", "text")),
                    metadata=mv,
                )
            )
        return results
