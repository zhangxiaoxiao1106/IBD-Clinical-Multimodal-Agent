"""系统里用到的 Pydantic 数据模型。"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


def _short_id() -> str:
    return uuid4().hex[:8]


class ModalityType(str, Enum):
    TABLE = "table"
    IMAGE = "image"
    FUSION = "fusion"


class TriageLevel(str, Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class AgentRole(str, Enum):
    DIRECTOR = "DirectorAgent"
    RETRIEVAL = "RetrievalAgent"
    ANALYSIS = "AnalysisAgent"
    REPORT = "ReportAgent"
    JUDGE = "JudgeAgent"


class MessageType(str, Enum):
    REQUEST = "request"
    COMMAND = "command"
    EVENT = "event"
    RESPONSE = "response"
    ERROR = "error"


class EvidenceLevel(str, Enum):
    RCT = "I级：随机对照试验 (RCT)"
    COHORT = "II级：队列研究"
    CASE_CONTROL = "III级：病例对照研究"
    CASE_REPORT = "IV级：病例报告 / 专家意见"


class AgentMessage(BaseModel):
    """Agent 之间传递的消息。trace 字段记录经过的节点。"""

    message_id: str = Field(default_factory=_short_id)
    correlation_id: str
    message_type: MessageType = MessageType.REQUEST
    sender: AgentRole
    receiver: Optional[AgentRole] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    payload: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    trace: List[str] = Field(default_factory=list)


class PatientInput(BaseModel):
    patient_id: str
    modality: ModalityType

    clinical_table: Optional[Dict[str, Any]] = None
    endoscopy_images: Optional[List[str]] = None
    pathology_images: Optional[List[str]] = None
    questionnaire: Optional[str] = None
    clinical_notes: Optional[str] = None

    @model_validator(mode="after")
    def _check_modality(self) -> "PatientInput":
        # 注意必须用 after 模式，否则 clinical_table 还没校验完，拿不到值
        if self.modality == ModalityType.TABLE and not self.clinical_table:
            raise ValueError("表格模态必须提供 clinical_table")
        if self.modality == ModalityType.IMAGE and not (
            self.endoscopy_images or self.pathology_images
        ):
            raise ValueError("图像模态必须提供至少一种图像")
        if self.modality == ModalityType.FUSION and not self.clinical_table:
            raise ValueError("融合模态必须以 clinical_table 为基础")
        return self


class EvidenceChunk(BaseModel):
    chunk_id: str = Field(default_factory=_short_id)
    trace_id: Optional[str] = None
    source_type: Literal["pubmed", "vector_db"]
    source_id: str
    source_title: str
    publication_year: Optional[int] = None
    evidence_level: EvidenceLevel = EvidenceLevel.CASE_REPORT
    chunk_text: str

    dense_score: Optional[float] = None
    fused_score: Optional[float] = None
    rerank_score: Optional[float] = None

    query_text: str
    retrieval_method: str
    retrieved_at: datetime = Field(default_factory=datetime.now)


class EvidenceChain(BaseModel):
    chain_id: str = Field(default_factory=_short_id)
    original_query: str
    expanded_queries: List[str] = Field(default_factory=list)
    evidence_list: List[EvidenceChunk] = Field(default_factory=list)

    # 检索阶段如果降级（比如向量库不可用），在这里说明，报告会据此提示证据局限
    notes: List[str] = Field(default_factory=list)

    def level_distribution(self) -> Dict[str, int]:
        dist: Dict[str, int] = {}
        for ev in self.evidence_list:
            key = ev.evidence_level.value
            dist[key] = dist.get(key, 0) + 1
        return dist

    def high_quality(self, min_level: EvidenceLevel = EvidenceLevel.COHORT) -> List[EvidenceChunk]:
        order = {
            EvidenceLevel.RCT: 4,
            EvidenceLevel.COHORT: 3,
            EvidenceLevel.CASE_CONTROL: 2,
            EvidenceLevel.CASE_REPORT: 1,
        }
        threshold = order[min_level]
        return [e for e in self.evidence_list if order[e.evidence_level] >= threshold]


SECTION_NAMES = ("background", "methods", "results", "evidence", "conclusion", "recommendation")


class ReportSection(BaseModel):
    section_name: Literal["background", "methods", "results", "evidence", "conclusion", "recommendation"]
    title: str
    content: str
    citations: List[str] = Field(default_factory=list)
    confidence: float = Field(0.5, ge=0.0, le=1.0)


class ClinicalReport(BaseModel):
    report_id: str = Field(default_factory=_short_id)
    patient_id: str
    generated_at: datetime = Field(default_factory=datetime.now)
    sections: List[ReportSection] = Field(default_factory=list)
    generated_by: str = "template"  # template / llm

    overall_confidence: float = Field(0.0, ge=0.0, le=1.0)
    evidence_coverage: float = Field(0.0, ge=0.0, le=1.0)
    hallucination_risk: float = Field(0.0, ge=0.0, le=1.0)
    rule_warnings: List[str] = Field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [f"# IBD 临床评估报告（{self.patient_id}）", ""]
        for sec in self.sections:
            lines.append(f"## {sec.title}")
            lines.append(sec.content)
            if sec.citations:
                lines.append(f"\n引用: {', '.join(sec.citations)}")
            lines.append("")
        lines.append(
            f"整体置信度: {self.overall_confidence:.2f} | "
            f"证据覆盖率: {self.evidence_coverage:.2f} | "
            f"幻觉风险: {self.hallucination_risk:.2f}"
        )
        return "\n".join(lines)


class JudgeDimension(BaseModel):
    dimension_name: str
    score: float = Field(..., ge=0.0, le=1.0)
    explanation: str
    suggestions: List[str] = Field(default_factory=list)


class JudgeResult(BaseModel):
    evaluation_id: str = Field(default_factory=_short_id)
    report_id: str
    evaluated_at: datetime = Field(default_factory=datetime.now)
    dimensions: List[JudgeDimension] = Field(default_factory=list)
    overall_score: float = Field(0.0, ge=0.0, le=1.0)
    passed: bool = False
    feedback: str = ""
    quality_metrics: Dict[str, float] = Field(default_factory=dict)
