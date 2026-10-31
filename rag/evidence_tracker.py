"""证据溯源：记录每条证据来自哪个查询、哪种检索方式，并对文本做哈希。

检索 Agent 每跑完一轮就登记一次，最后可以导出审计报告。
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

from models.schemas import EvidenceChain


@dataclass
class EvidenceTrace:
    trace_id: str
    source_id: str
    source_title: str
    query_text: str
    retrieval_method: str
    evidence_level: str
    chunk_hash: str
    cited_in_sections: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


class EvidenceTracker:
    def __init__(self) -> None:
        self._traces: Dict[str, EvidenceTrace] = {}

    @staticmethod
    def _hash(text: str) -> str:
        # 只存前 16 位，够做完整性校验即可
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def register_chain(self, chain: EvidenceChain) -> None:
        for i, chunk in enumerate(chain.evidence_list):
            trace_id = f"tr_{chunk.source_id or i}_{i}"
            chunk.trace_id = trace_id
            self._traces[trace_id] = EvidenceTrace(
                trace_id=trace_id,
                source_id=chunk.source_id,
                source_title=chunk.source_title,
                query_text=chunk.query_text,
                retrieval_method=chunk.retrieval_method,
                evidence_level=chunk.evidence_level.value,
                chunk_hash=self._hash(chunk.chunk_text),
            )

    def mark_cited(self, trace_id: str, section: str) -> None:
        trace = self._traces.get(trace_id)
        if trace and section not in trace.cited_in_sections:
            trace.cited_in_sections.append(section)

    def verify(self, trace_id: str, text: str) -> bool:
        trace = self._traces.get(trace_id)
        return trace is not None and trace.chunk_hash == self._hash(text)

    def audit_report(self) -> Dict[str, Any]:
        method_dist: Dict[str, int] = {}
        for t in self._traces.values():
            method_dist[t.retrieval_method] = method_dist.get(t.retrieval_method, 0) + 1
        cited = sum(1 for t in self._traces.values() if t.cited_in_sections)
        return {
            "total_evidence": len(self._traces),
            "cited_in_report": cited,
            "retrieval_method_distribution": method_dist,
        }
