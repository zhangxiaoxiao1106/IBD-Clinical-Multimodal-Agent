"""裁判 Agent：从事实一致性、引用准确性、临床完整性、逻辑连贯性四个维度打分。

事实一致性调用 NLI 工具，NLI 不可用时该维度按“无法校验”给中性分并在说明里标注。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from models.schemas import (
    AgentMessage, AgentRole, ClinicalReport, EvidenceChain, JudgeDimension, JudgeResult,
)
from core.agent_base import AgentBase
from evaluation import quality_metrics
from config.settings import agent_config

ACTION_WORDS = ("建议", "推荐", "考虑", "应", "需")
POSITIVE_WORDS = ("良好", "有效", "缓解", "达标", "低危")
AGGRESSIVE_WORDS = ("升级", "换药", "加强", "手术")


class JudgeAgent(AgentBase):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(AgentRole.JUDGE, **kwargs)

    async def on_init(self) -> None:
        self.pass_score = agent_config().get("judge", {}).get("scoring", {}).get("min_pass_score", 0.8)

    async def on_message(self, message: AgentMessage) -> Optional[Dict[str, Any]]:
        report_data = message.payload.get("report") or {}
        chain_data = message.payload.get("evidence_chain") or {}
        if not report_data:
            return {"error": "裁判 Agent 没有收到报告"}
        report = ClinicalReport(**report_data)
        chain = EvidenceChain(**chain_data) if chain_data else None

        dimensions = [
            await self._factual_consistency(report, chain),
            self._citation_accuracy(report, chain),
            self._completeness(report),
            self._coherence(report),
        ]
        overall = sum(d.score for d in dimensions) / len(dimensions)

        answer = "\n".join(s.content for s in report.sections if s.section_name
                           in ("results", "conclusion", "recommendation"))
        contexts = [ev.chunk_text for ev in chain.evidence_list] if chain else []
        metrics = quality_metrics.calculate_all(
            question="IBD 抗 TNF 治疗预后如何", answer=answer, contexts=contexts
        )

        result = JudgeResult(
            report_id=report.report_id,
            dimensions=dimensions,
            overall_score=round(overall, 3),
            passed=overall >= self.pass_score,
            quality_metrics=metrics,
        )
        if not result.passed:
            weak = [d.dimension_name for d in dimensions if d.score < self.pass_score]
            result.feedback = f"以下维度未达标: {', '.join(weak)}"
        return {"judge_result": result.model_dump()}

    async def _factual_consistency(self, report: ClinicalReport,
                                   chain: Optional[EvidenceChain]) -> JudgeDimension:
        if not chain or not chain.evidence_list:
            return JudgeDimension(
                dimension_name="factual_consistency", score=0.5,
                explanation="没有检索证据，无法做事实一致性校验",
                suggestions=["在有网络/向量库的环境下重新运行检索"],
            )

        sentences = [s.strip() for sec in report.sections
                     for s in sec.content.split("。") if len(s.strip()) > 20]
        entail, contradiction, checked = 0, 0, 0
        nli_available = True
        for sentence in sentences[:3]:
            evidence = chain.evidence_list[0]
            resp = await self.call_tool("nli_validate", {
                "premise": evidence.chunk_text[:500],
                "hypothesis": sentence[:500],
            })
            if not resp.get("available", True):
                nli_available = False
                continue
            checked += 1
            if resp["label"] == "entailment":
                entail += 1
            elif resp["label"] == "contradiction":
                contradiction += 1

        if checked == 0:
            return JudgeDimension(
                dimension_name="factual_consistency", score=0.5,
                explanation="NLI 模型不可用，未完成事实校验",
                suggestions=["安装 transformers/torch 后重跑"],
            )
        score = entail / checked
        note = "NLI 已校验" if nli_available else "部分断言 NLI 未校验"
        return JudgeDimension(
            dimension_name="factual_consistency", score=score,
            explanation=f"{note}：蕴含 {entail}，矛盾 {contradiction}，共 {checked} 条断言",
            suggestions=["存在矛盾断言，需回溯证据"] if contradiction else [],
        )

    @staticmethod
    def _citation_accuracy(report: ClinicalReport,
                           chain: Optional[EvidenceChain]) -> JudgeDimension:
        report_pmids = {pmid for sec in report.sections for pmid in sec.citations}
        if not chain:
            return JudgeDimension(
                dimension_name="citation_accuracy", score=0.3,
                explanation="无证据链，无法核对引用",
            )
        evidence_pmids = {ev.source_id for ev in chain.evidence_list if ev.source_id}
        if not report_pmids:
            return JudgeDimension(
                dimension_name="citation_accuracy", score=0.4,
                explanation="报告中没有 PMID 引用",
                suggestions=["关键结论后补充 PMID 引用"],
            )
        valid = report_pmids & evidence_pmids
        score = len(valid) / len(report_pmids)
        return JudgeDimension(
            dimension_name="citation_accuracy", score=score,
            explanation=f"引用 {len(report_pmids)} 条，其中 {len(valid)} 条在证据链内",
            suggestions=["移除不在本次检索结果中的引用"] if len(valid) < len(report_pmids) else [],
        )

    @staticmethod
    def _completeness(report: ClinicalReport) -> JudgeDimension:
        required = {"background", "methods", "results", "evidence", "conclusion", "recommendation"}
        have = {s.section_name for s in report.sections}
        missing = required - have
        score = 1.0 - 0.15 * len(missing)
        rec = next((s for s in report.sections if s.section_name == "recommendation"), None)
        if rec and not any(w in rec.content for w in ACTION_WORDS):
            score -= 0.15
        return JudgeDimension(
            dimension_name="clinical_completeness", score=max(score, 0.0),
            explanation=f"缺失章节: {sorted(missing)}" if missing else "六段结构完整",
            suggestions=[f"补充 {m}" for m in missing],
        )

    @staticmethod
    def _coherence(report: ClinicalReport) -> JudgeDimension:
        def content(name: str) -> str:
            sec = next((s for s in report.sections if s.section_name == name), None)
            return sec.content if sec else ""

        conclusion, recommendation = content("conclusion"), content("recommendation")
        contradiction = (
            any(w in conclusion for w in POSITIVE_WORDS)
            and any(w in recommendation for w in AGGRESSIVE_WORDS)
        )
        score = 0.75 if contradiction else 1.0
        return JudgeDimension(
            dimension_name="logical_coherence", score=score,
            explanation="结论偏乐观但建议偏激进，存在矛盾" if contradiction else "未发现明显矛盾",
            suggestions=["核对结论与建议是否一致"] if contradiction else [],
        )

    async def on_shutdown(self) -> None:
        return None
