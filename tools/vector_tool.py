"""FAISS 向量检索。

faiss 和 sentence-transformers 是可选依赖：
- 没装或索引/模型不存在时，工具返回 available=False 的空结果（由上层走降级），
  不会在启动时下载模型。
- 索引需要先通过 add_documents 构建并 save_index 落盘，本项目只负责读取。
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

from core.tool_registry import BaseTool, ToolSchema
from tools.base import VectorSearchInput, VectorSearchOutput


class VectorSearchTool(BaseTool):
    def __init__(self, index_path: str = "./data/faiss_index.bin",
                 embedding_model: str = "BAAI/bge-large-en-v1.5"):
        super().__init__(ToolSchema(
            name="vector_search",
            description="本地 FAISS 向量检索（依赖缺失时返回空结果）",
            input_schema=VectorSearchInput,
            output_schema=VectorSearchOutput,
            timeout_seconds=20.0,
            max_retries=0,
            fallback_value={"results": [], "available": False,
                            "reason": "向量检索执行失败", "query_embedding_dim": 0,
                            "search_time_ms": 0.0},
        ))
        self.index_path = index_path
        self.embedding_model_name = embedding_model
        self._encoder = None
        self._faiss = None
        self._index = None
        self._metadata: List[Dict[str, Any]] = []
        self._ready = False

    def _ensure_loaded(self) -> None:
        if self._ready:
            return
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(f"缺少向量检索依赖: {exc}")

        self._faiss = faiss
        if os.path.exists(self.index_path):
            self._index = faiss.read_index(self.index_path)
            meta_path = self.index_path + ".meta"
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    self._metadata = json.load(f)
        else:
            # 没有索引文件就给一个空索引，检索自然返回空
            self._index = faiss.IndexFlatIP(1024)
        self._encoder = SentenceTransformer(self.embedding_model_name)
        self._ready = True

    async def _execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        try:
            self._ensure_loaded()
        except RuntimeError as exc:
            return {"results": [], "available": False, "query_embedding_dim": 0,
                    "reason": str(exc), "search_time_ms": 0.0}

        params = VectorSearchInput(**inputs)
        start = time.time()

        if self._index is None or self._index.ntotal == 0:
            return {"results": [], "available": True, "query_embedding_dim": 0,
                    "reason": "索引为空", "search_time_ms": (time.time() - start) * 1000}

        import numpy as np
        vector = self._encoder.encode(
            params.query_text, normalize_embeddings=True, convert_to_numpy=True
        )
        vector = np.asarray([vector], dtype="float32")
        distances, indices = self._index.search(vector, params.top_k)

        results: List[Dict[str, Any]] = []
        for idx, score in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue
            meta = self._metadata[idx]
            results.append({
                "source_id": meta.get("source_id", ""),
                "source_title": meta.get("source_title", ""),
                "chunk_text": meta.get("chunk_text", ""),
                "evidence_level": meta.get("evidence_level", "IV级：病例报告 / 专家意见"),
                "score": float(score),
                "year": meta.get("year"),
            })

        return {
            "results": results,
            "available": True,
            "query_embedding_dim": int(vector.shape[1]),
            "reason": "",
            "search_time_ms": (time.time() - start) * 1000,
        }

    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> None:
        """批量建索引（离线准备数据时使用，演示流程不调用）。"""
        self._ensure_loaded()
        import numpy as np
        embeddings = self._encoder.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )
        self._index.add(np.asarray(embeddings, dtype="float32"))
        self._metadata.extend(metadatas)

    def save_index(self) -> None:
        if self._index is None:
            return
        os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)
        self._faiss.write_index(self._index, self.index_path)
        with open(self.index_path + ".meta", "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, ensure_ascii=False)
