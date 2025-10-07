from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import chromadb

from .models import ClipMultiModal
from .ingestion import Record


class MultiModalIndex:
    """
    Chroma-based unified index for text, image and audio embeddings in a shared CLIP space.
    Supports both persistent (on-disk) and ephemeral (in-memory) modes.
    """

    def __init__(self, persist_dir: Path | None = None, persistent: bool = False):
        self.persistent = persistent
        if persistent:
            if persist_dir is None:
                raise ValueError("persist_dir is required when persistent=True")
            self.persist_dir = Path(persist_dir)
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        else:
            self.persist_dir = None
            self.client = chromadb.EphemeralClient()
        self.collection = self.client.get_or_create_collection(
            name="multimodal",
        )

    def add_records(self, records: List[Record]):
        if not records:
            return
        ids = [r.id for r in records]
        embeddings = [r.embedding for r in records]
        metadatas = [r.metadata | {"modality": r.modality} for r in records]
        documents = [r.metadata.get("text", "") for r in records]  # optional
        self.collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)

    def query(self, query_embedding: List[float], k: int = 6):
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["metadatas", "distances", "documents"],
        )

    def count(self) -> int:
        return self.collection.count()

    def reset(self):
        # Delete and recreate empty collection
        try:
            self.client.delete_collection("multimodal")
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(name="multimodal")
