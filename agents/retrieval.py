"""检索 Agent：生成查询 -> 并行查 PubMed 和本地向量库 -> 融合排序 -> 建证据链。

没有向量库时只靠 PubMed 也能跑；融合用 RRF（按排名融合），
最后按证据等级做一次启发式重排，不是 cross-encoder。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

from models.schemas import AgentMessage, AgentRole, EvidenceChain, EvidenceChunk, EvidenceLevel
from core.agent_base import AgentBase
from rag.query_generator import MedicalQueryGenerator
from rag.evidence_tracker import EvidenceTracker
from config.settings import rag_config


def to_level(raw: str) -> EvidenceLevel:
    try:
        return EvidenceLevel(raw)
    except ValueError:
        return EvidenceLevel.CASE_REPORT


LEVEL_BONUS = {
    EvidenceLevel.RCT: 0.15,
    EvidenceLevel.COHORT: 0.10,
    EvidenceLevel.CASE_CONTROL: 0.05,
    EvidenceLevel.CASE_REPORT: 0.0,
}


class RetrievalAgent(AgentBase):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(AgentRole.RETRIEVAL, **kwargs)
        self.generator = MedicalQueryGenerator()
        self.tracker = EvidenceTracker()

    async def on_init(self) -> None:
        return None

    async def on_message(self, message: AgentMessage) -> Optional[Dict[str, Any]]:
        patient = message.payload.get("patient_input") or {}
        cfg = rag_config()
        date_range = cfg.get("pubmed", {}).get("date_range", "2020:2026")
        top_k = cfg.get("fusion", {}).get("top_k_final", 15)

        clinical_text = self._build_clinical_text(patient)
        queries = self.generator.generate_queries(clinical_text)

        # 精确式检索为 0 时，退到扩展查询放宽条件
        pubmed_resp = {"articles": []}
        used_query = queries[0]
        for query in queries[:2]:
            pubmed_resp = await self.call_tool("pubmed_search", {
                "query": query, "max_results": 15, "date_range": date_range,
            })
            if pubmed_resp.get("articles"):
                used_query = query
                break
        vector_resp = await self.call_tool("vector_search", {
            "query_text": queries[0], "top_k": 20,
        })

        chunks: List[EvidenceChunk] = []
        notes: List[str] = []

        for art in pubmed_resp.get("articles", [])[:20]:
            if not art.get("abstract"):
                continue
            chunks.append(EvidenceChunk(
                source_type="pubmed",
                source_id=art.get("pmid", ""),
                source_title=art.get("title", ""),
                publication_year=art.get("year"),
                evidence_level=to_level(art.get("evidence_level", "")),
                chunk_text=art["abstract"][:800],
                query_text=used_query,
                retrieval_method="pubmed_api",
            ))
        if self.tools.get("pubmed_search").health == "degraded":
            notes.append("PubMed 检索不可用，未纳入文献证据")
        elif not chunks:
            notes.append(f"PubMed 未检索到含摘要的文献（查询式: {used_query}）")

        for item in vector_resp.get("results", [])[:20]:
            chunks.append(EvidenceChunk(
                source_type="vector_db",
                source_id=item.get("source_id", ""),
                source_title=item.get("source_title", ""),
                evidence_level=to_level(item.get("evidence_level", "")),
                chunk_text=item.get("chunk_text", ""),
                dense_score=item.get("score"),
                query_text=queries[0],
                retrieval_method="dense_vector",
            ))
        if not vector_resp.get("available", True):
            notes.append(f"向量检索跳过: {vector_resp.get('reason', '不可用')}")

        fused = self._rrf_fuse(chunks)
        reranked = self._heuristic_rerank(fused)

        chain = EvidenceChain(
            original_query=used_query,
            expanded_queries=queries[1:],
            evidence_list=reranked[:top_k],
            notes=notes,
        )
        self.tracker.register_chain(chain)
        return {
            "evidence_chain": chain.model_dump(),
            "evidence_level_distribution": chain.level_distribution(),
            "audit": self.tracker.audit_report(),
        }

    @staticmethod
    def _build_clinical_text(patient: Dict[str, Any]) -> str:
        table = patient.get("clinical_table") or {}
        # 临床表里存的是缩写，补成 PubMed 能检索到的全称
        disease_full = {"CD": "Crohn disease", "UC": "ulcerative colitis"}
        disease = disease_full.get(table.get("disease_type"), str(table.get("disease_type", "")))
        parts = [
            disease,
            str(table.get("current_biologic", "")),
            " ".join(table.get("prior_biologics", []) or []),
            " ".join(table.get("candidate_genes", []) or []),
            patient.get("questionnaire") or "",
        ]
        return " ".join(p for p in parts if p)

    @staticmethod
    def _rrf_fuse(chunks: List[EvidenceChunk], k: int = 60) -> List[EvidenceChunk]:
        """按来源分别排名，同一 source_id 融合，分数为 RRF 之和。"""
        by_method: Dict[str, List[EvidenceChunk]] = defaultdict(list)
        for chunk in chunks:
            by_method[chunk.retrieval_method].append(chunk)

        scores: Dict[str, float] = defaultdict(float)
        merged: Dict[str, EvidenceChunk] = {}
        for group in by_method.values():
            for rank, chunk in enumerate(group, start=1):
                key = chunk.source_id or chunk.chunk_id
                scores[key] += 1.0 / (k + rank)
                # 同一文献两路都命中时保留信息更全的那条
                if key not in merged or len(chunk.chunk_text) > len(merged[key].chunk_text):
                    merged[key] = chunk

        ordered = sorted(merged.values(), key=lambda c: scores[c.source_id or c.chunk_id],
                         reverse=True)
        for chunk in ordered:
            chunk.fused_score = round(scores[chunk.source_id or chunk.chunk_id], 5)
        return ordered

    @staticmethod
    def _heuristic_rerank(chunks: List[EvidenceChunk]) -> List[EvidenceChunk]:
        """RRF 分数基础上加证据等级权重，作为最终排序。"""
        for chunk in chunks:
            chunk.rerank_score = round((chunk.fused_score or 0.0) + LEVEL_BONUS[chunk.evidence_level], 5)
        return sorted(chunks, key=lambda c: c.rerank_score or 0.0, reverse=True)

    async def on_shutdown(self) -> None:
        return None
