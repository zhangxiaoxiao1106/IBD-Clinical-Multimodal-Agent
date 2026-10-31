"""报告 Agent：生成六段式结构化报告。

有 LLM 服务时解析 LLM 的 JSON 输出；服务不可用或解析失败时，
用分析结果和检索证据直接拼模板报告（这是默认演示路径，离线可跑）。
报告生成后统一过一遍规则层幻觉检查。
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from models.schemas import (
    AgentMessage, AgentRole, ClinicalReport, EvidenceChain, ReportSection,
)
from core.agent_base import AgentBase
from evaluation.hallucination_detector import HallucinationDetector

SECTION_TITLES = {
    "background": "背景综述",
    "methods": "评估方法",
    "results": "分析结果",
    "evidence": "证据汇总",
    "conclusion": "临床结论",
    "recommendation": "用药建议",
}


class ReportAgent(AgentBase):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(AgentRole.REPORT, **kwargs)
        self.detector = HallucinationDetector()

    async def on_init(self) -> None:
        return None

    async def on_message(self, message: AgentMessage) -> Optional[Dict[str, Any]]:
        analysis = message.payload.get("analysis_result") or {}
        chain_data = message.payload.get("evidence_chain") or {}
        chain = EvidenceChain(**chain_data) if chain_data else None

        report = self._template_report(analysis, chain)
        report = self._finalize(report)

        # 有 LLM 服务时尝试用 LLM 重写并替换模板；失败就保留模板，不影响结果
        llm_report = await self._try_llm_report(analysis, chain)
        if llm_report is not None:
            llm_report = self._finalize(llm_report)
            if llm_report.overall_confidence >= report.overall_confidence:
                report = llm_report

        return {"report": report.model_dump()}

    # ---------- 模板生成 ----------

    def _template_report(self, analysis: Dict[str, Any],
                         chain: Optional[EvidenceChain]) -> ClinicalReport:
        features = analysis.get("features", {})
        risk = analysis.get("risk_assessment", {})
        summary = analysis.get("evidence_summary", {})
        top = summary.get("top_findings", [])
        pmids = [f["source_id"] for f in top if f.get("source_id")]
        primary_cite = [f"PMID:{pmids[0]}"] if pmids else []

        bg = (
            f"患者 {features.get('patient_id', 'unknown')}，{features.get('age', '年龄不详')} 岁，"
            f"诊断为 {features.get('disease_type', '待明确')}，"
            f"病程 {features.get('disease_duration_months', '不详')} 个月。"
            f"当前使用 {features.get('current_biologic') or '无生物制剂'}；"
            f"既往生物制剂: {', '.join(features.get('prior_biologics', [])) or '无'}。"
        )

        methods = (
            f"本次评估基于患者临床表格指标与问卷文本，检索 PubMed 文献共 "
            f"{summary.get('total', 0)} 篇（其中 II 级及以上 {summary.get('high_quality', 0)} 篇），"
            "结合 CDAI/Mayo、钙卫蛋白等指标进行规则风险分层。"
        )

        reason_text = "；".join(risk.get("reasons", [])) or "未发现明显高危因素"
        results = (
            f"风险评分 {risk.get('risk_score', 0.0)}，风险等级 {risk.get('risk_level', 'unknown')}。"
            f"主要依据：{reason_text}。"
            + (f" 相关证据见 {primary_cite[0]}。" if primary_cite else "")
        )

        if top:
            evidence_lines = [
                f"- [{f['level']}] {f['title']}（PMID:{f['source_id']}）: {f['snippet'][:120]}"
                for f in top
            ]
            evidence_text = "\n".join(evidence_lines)
        else:
            evidence_text = "本次未检索到可用文献，以下结论主要依据患者临床指标，证据有限，建议进一步验证。"

        conclusion = (
            f"现有资料提示患者总体风险为 {risk.get('risk_level', 'unknown')} 等"
            f"（置信度 {risk.get('confidence', 0.0)}）。"
            + (f" 文献证据支持上述判断，见 {primary_cite[0]}。" if primary_cite
               else " 因缺少外部文献，结论需结合临床随访确认。")
        )

        recommendation = (
            f"{risk.get('advice', '建议结合临床综合判断')}。"
            "若症状加重或炎症指标升高，应及时复查内镜与钙卫蛋白。"
        )

        contents = {
            "background": bg,
            "methods": methods,
            "results": results,
            "evidence": evidence_text,
            "conclusion": conclusion,
            "recommendation": recommendation,
        }
        sections: List[ReportSection] = []
        for key in ("background", "methods", "results", "evidence", "conclusion", "recommendation"):
            refs = re.findall(r"PMID:(\d+)", contents[key])
            sections.append(ReportSection(
                section_name=key,
                title=SECTION_TITLES[key],
                content=contents[key],
                citations=refs,
                confidence=0.7 if refs else 0.5,
            ))

        return ClinicalReport(
            patient_id=features.get("patient_id", analysis.get("patient_id", "unknown")),
            sections=sections,
            generated_by="template",
        )

    # ---------- LLM 生成 ----------

    def _build_prompt(self, analysis: Dict[str, Any], chain: Optional[EvidenceChain]) -> str:
        features = analysis.get("features", {})
        risk = analysis.get("risk_assessment", {})
        evidence = ""
        if chain:
            evidence = "\n".join(
                f"- PMID:{ev.source_id}（{ev.evidence_level.value}）: {ev.chunk_text[:200]}"
                for ev in chain.evidence_list[:5]
            )
        return f"""请基于以下信息生成 IBD 抗 TNF 治疗预后评估报告，严格输出 JSON，
键为 background/methods/results/evidence/conclusion/recommendation，值为字符串。
结论必须使用"提示""建议"等审慎措辞，每个关键结论后用 [PMID:xxxx] 标注引用。

患者: {features.get('age')} 岁, {features.get('disease_type')},
当前用药 {features.get('current_biologic')}, 既往 {features.get('prior_biologics')}
风险: {risk.get('risk_level')} ({risk.get('risk_score')}), 依据 {risk.get('reasons')}
证据:
{evidence or '本次无检索证据，请在报告中明确说明证据有限'}
"""

    async def _try_llm_report(self, analysis: Dict[str, Any],
                              chain: Optional[EvidenceChain]) -> Optional[ClinicalReport]:
        try:
            resp = await self.call_tool("llm_generate", {
                "prompt": self._build_prompt(analysis, chain),
                "temperature": 0.3,
                "max_tokens": 2500,
            })
        except Exception:
            return None
        if resp.get("finish_reason") == "unavailable" or not resp.get("generated_text"):
            return None

        match = re.search(r"\{.*\}", resp["generated_text"], re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return None

        sections = []
        for key, title in SECTION_TITLES.items():
            content = data.get(key, "")
            if not content:
                return None
            refs = re.findall(r"\[?PMID:(\d+)\]?", content)
            sections.append(ReportSection(
                section_name=key, title=title, content=content,
                citations=refs, confidence=0.8 if refs else 0.5,
            ))
        patient_id = analysis.get("features", {}).get("patient_id", "unknown")
        return ClinicalReport(patient_id=patient_id, sections=sections, generated_by="llm")

    # ---------- 后处理 ----------

    def _finalize(self, report: ClinicalReport) -> ClinicalReport:
        full_text = "\n".join(s.content for s in report.sections)
        risk, warnings = self.detector.rule_check(full_text)
        report.rule_warnings = warnings
        report.hallucination_risk = round(risk, 3)

        cited = sum(1 for s in report.sections if s.citations)
        report.evidence_coverage = round(cited / len(report.sections), 3) if report.sections else 0.0

        report.overall_confidence = round(
            (sum(s.confidence for s in report.sections) / len(report.sections)
             if report.sections else 0.5) - 0.2 * risk,
            3,
        )
        return report

    async def on_shutdown(self) -> None:
        return None
