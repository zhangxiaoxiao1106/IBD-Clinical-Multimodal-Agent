"""报告模板生成与幻觉规则检查的测试。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.report import ReportAgent
from evaluation.hallucination_detector import HallucinationDetector
from evaluation import quality_metrics


def _analysis():
    return {
        "patient_id": "p1",
        "features": {
            "patient_id": "p1",
            "age": 30,
            "disease_type": "CD",
            "disease_duration_months": 12,
            "current_biologic": "infliximab",
            "prior_biologics": [],
            "calprotectin": 100,
            "cdai": 150,
        },
        "evidence_summary": {"total": 0, "high_quality": 0, "top_findings": [], "notes": []},
        "risk_assessment": {
            "risk_score": 0.5,
            "risk_level": "moderate",
            "confidence": 0.6,
            "advice": "6-8 周后复查",
            "reasons": ["测试依据"],
        },
    }


def test_template_report_has_six_sections():
    agent = ReportAgent()
    report = agent._template_report(_analysis(), None)
    report = agent._finalize(report)
    assert len(report.sections) == 6
    assert [s.section_name for s in report.sections] == [
        "background", "methods", "results", "evidence", "conclusion", "recommendation"
    ]
    assert report.generated_by == "template"
    # 无证据时必须说明证据有限
    assert "证据有限" in report.sections[3].content


def test_absolute_claim_flagged():
    detector = HallucinationDetector()
    risk, issues = detector.rule_check("该药物必然治愈所有患者，100% 有效。")
    assert risk > 0
    assert issues


def test_chinese_token_overlap():
    metrics = quality_metrics.calculate_all(
        question="患者是否需要升级治疗",
        answer="患者目前风险较低，建议维持治疗并随访",
        contexts=["患者目前风险较低，建议维持治疗，定期随访"],
    )
    assert 0.0 <= metrics["faithfulness"] <= 1.0
    assert metrics["faithfulness"] > 0.5
