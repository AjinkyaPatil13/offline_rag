from __future__ import annotations

from typing import Any, Dict, List


def format_citation(idx: int, meta: Dict[str, Any]) -> str:
    src = meta.get("source_name") or meta.get("source_path", "source")
    modality = meta.get("modality", meta.get("type", "text"))
    if modality == "text":
        page = meta.get("page")
        chunk = meta.get("chunk")
        if page:
            return f"[{idx}] {src} (page {page}, chunk {chunk})"
        return f"[{idx}] {src} (chunk {chunk})"
    if modality == "image":
        return f"[{idx}] {src} (image)"
    if modality == "audio":
        st = meta.get("start")
        en = meta.get("end")
        return f"[{idx}] {src} (audio {st:.1f}-{en:.1f}s)"
    return f"[{idx}] {src}"


def build_context_with_citations(results: List[Dict[str, Any]], max_chars: int = 4000) -> str:
    """
    Build a text context for the LLM with numbered citations.
    Only textual content is included in the context body; images are described, audio uses transcripts.
    """
    parts: List[str] = []
    for i, r in enumerate(results, start=1):
        meta = r["metadata"]
        modality = meta.get("modality", meta.get("type", "text"))
        if modality == "image":
            desc = f"[IMAGE] {meta.get('source_name')} at {meta.get('image_path')}"
        else:
            desc = meta.get("text", "")
        citation = format_citation(i, meta)
        snippet = f"(Citation {i}: {citation})\n{desc}\n"
        parts.append(snippet)
        if sum(len(p) for p in parts) > max_chars:
            break
    return "\n\n".join(parts)
