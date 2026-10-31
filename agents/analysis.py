"""分析 Agent：把患者临床表格和检索证据整理成结构化特征，再做规则风险分层。

没有真正的多模态模型，图像/组学部分目前只判断“有没有”，
风险评分基于 CDAI/Mayo/钙卫蛋白/既往生物制剂失败史等明确规则。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from models.schemas import AgentMessage, AgentRole, EvidenceChain, EvidenceLevel
from core.agent_base import AgentBase


class AnalysisAgent(AgentBase):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(AgentRole.ANALYSIS, **kwargs)

    async def on_init(self) -> None:
        return None

    async def on_message(self, message: AgentMessage) -> Optional[Dict[str, Any]]:
        patient = message.payload.get("patient_input") or {}
        chain_data = message.payload.get("evidence_chain") or {}
        chain = EvidenceChain(**chain_data) if chain_data else None

        features = self._extract_features(patient)
        evidence_summary = self._summarize(chain)
        risk = self._risk_stratification(features, chain)

        return {
            "analysis_result": {
                "patient_id": features["patient_id"],
                "features": features,
                "evidence_summary": evidence_summary,
                "risk_assessment": risk,
            }
        }

    @staticmethod
    def _extract_features(patient: Dict[str, Any]) -> Dict[str, Any]:
        table = patient.get("clinical_table") or {}
        return {
            "patient_id": patient.get("patient_id", "unknown"),
            "age": table.get("age"),
            "disease_type": table.get("disease_type", "未知"),
            "disease_duration_months": table.get("disease_duration_months"),
            "current_biologic": table.get("current_biologic"),
            "prior_biologics": table.get("prior_biologics", []) or [],
            "immunomodulators": table.get("immunomodulators", []) or [],
            "cdai": table.get("cdai"),
            "mayo_score": table.get("mayo_score"),
            "calprotectin": table.get("calprotectin"),
            "crp": table.get("crp"),
            "candidate_genes": table.get("candidate_genes", []) or [],
            "has_imaging": bool(patient.get("endoscopy_images") or patient.get("pathology_images")),
        }

    @staticmethod
    def _summarize(chain: Optional[EvidenceChain]) -> Dict[str, Any]:
        if not chain or not chain.evidence_list:
            return {"total": 0, "high_quality": 0, "level_distribution": {},
                    "top_findings": [], "notes": chain.notes if chain else []}

        high = chain.high_quality(EvidenceLevel.COHORT)
        return {
            "total": len(chain.evidence_list),
            "high_quality": len(high),
            "level_distribution": chain.level_distribution(),
            "top_findings": [
                {
                    "source_id": ev.source_id,
                    "title": ev.source_title[:60],
                    "level": ev.evidence_level.value,
                    "trace_id": ev.trace_id,
                    "snippet": ev.chunk_text[:200],
                }
                for ev in chain.evidence_list[:5]
            ],
            "notes": chain.notes,
        }

    @staticmethod
    def _risk_stratification(features: Dict[str, Any],
                             chain: Optional[EvidenceChain]) -> Dict[str, Any]:
        score = 0.5
        reasons = []

        if (features.get("cdai") or 0) > 300:
            score += 0.12
            reasons.append(f"CDAI={features['cdai']}，提示中重度活动")
        if (features.get("mayo_score") or 0) >= 4:
            score += 0.12
            reasons.append(f"Mayo={features['mayo_score']}，提示中重度活动")
        if (features.get("calprotectin") or 0) > 250:
            score += 0.10
            reasons.append(f"钙卫蛋白={features['calprotectin']} μg/g，黏膜炎症活动")
        if len(features.get("prior_biologics", [])) >= 1:
            score += 0.10
            reasons.append("既往有生物制剂失败史，需警惕继发性失应答")

        evidence_count = len(chain.evidence_list) if chain else 0
        confidence = min(0.9, 0.55 + 0.03 * evidence_count)

        score = max(0.0, min(1.0, score))
        if score >= 0.7:
            level, advice = "high", "建议尽快复评，考虑升级治疗或更换作用机制的生物制剂"
        elif score >= 0.4:
            level, advice = "moderate", "建议优化当前方案并在 6-8 周后复查炎症指标"
        else:
            level, advice = "low", "当前方案反应尚可，建议维持治疗并定期随访"

        return {"risk_score": round(score, 3), "risk_level": level,
                "confidence": round(confidence, 2), "advice": advice, "reasons": reasons}
