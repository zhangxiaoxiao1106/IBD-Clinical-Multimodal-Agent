"""各工具的输入输出 Schema。"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class PubMedSearchInput(BaseModel):
    query: str
    max_results: int = Field(15, ge=1, le=100)
    date_range: Optional[str] = Field(None, description="形如 2020:2026")


class PubMedSearchOutput(BaseModel):
    total_found: int = 0
    returned: int = 0
    articles: List[dict] = Field(default_factory=list)
    query_used: str = ""
    search_time_ms: float = 0.0


class VectorSearchInput(BaseModel):
    query_text: str
    top_k: int = Field(20, ge=1, le=100)


class VectorSearchOutput(BaseModel):
    results: List[dict] = Field(default_factory=list)
    query_embedding_dim: int = 0
    available: bool = True
    reason: str = ""
    search_time_ms: float = 0.0


class LLMGenerateInput(BaseModel):
    prompt: str
    system_prompt: Optional[str] = None
    temperature: float = Field(0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(2048, ge=1, le=8192)


class LLMGenerateOutput(BaseModel):
    generated_text: str
    finish_reason: str
    total_tokens: int = 0
    generation_time_ms: float = 0.0


class NLIValidateInput(BaseModel):
    premise: str
    hypothesis: str
    model_name: str = "roberta-large-mnli"


class NLIValidateOutput(BaseModel):
    label: str  # entailment / contradiction / neutral
    confidence: float = Field(..., ge=0.0, le=1.0)
    available: bool = True
