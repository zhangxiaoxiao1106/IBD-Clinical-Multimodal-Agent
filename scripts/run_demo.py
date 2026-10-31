"""端到端演示：
患者输入 -> Director 分诊 -> Retrieval 检索 -> Analysis 分析 -> Report 报告 -> Judge 质检

    python scripts/run_demo.py

PubMed 需要外网；LLM/向量库/NLI 不可用时会自动降级，流程仍然跑通。
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.logging_setup import setup_logging
from core.memory import MemoryManager
from core.tool_registry import ToolRegistry
from models.schemas import AgentMessage, AgentRole
from agents.director import DirectorAgent
from agents.retrieval import RetrievalAgent
from agents.analysis import AnalysisAgent
from agents.report import ReportAgent
from agents.judge import JudgeAgent
from tools.pubmed_tool import PubMedTool
from tools.vector_tool import VectorSearchTool
from tools.llm_tool import LLMTool
from tools.nli_tool import NLITool


async def build_system() -> DirectorAgent:
    setup_logging()

    tools = ToolRegistry()
    tools.register(PubMedTool())
    tools.register(VectorSearchTool(index_path=str(PROJECT_ROOT / "data/faiss_index.bin")))
    tools.register(LLMTool())
    tools.register(NLITool())

    memory = MemoryManager()
    director = DirectorAgent(memory=memory, tools=tools)
    retrieval = RetrievalAgent(memory=memory, tools=tools)
    analysis = AnalysisAgent(memory=memory, tools=tools)
    report = ReportAgent(memory=memory, tools=tools)
    judge = JudgeAgent(memory=memory, tools=tools)

    # 互相注册下游路由，Director 才能调到各 Agent
    for agent in (retrieval, analysis, report, judge):
        director.register_peer(agent)
    for agent in (director, retrieval, analysis, report, judge):
        await agent.initialize()

    return director


async def run_case(director: DirectorAgent, case: dict) -> dict:
    print("=" * 60)
    print(f"病例 {case['patient_id']}")
    print("=" * 60)

    message = AgentMessage(
        correlation_id=f"case_{case['patient_id']}",
        sender=AgentRole.DIRECTOR,
        receiver=AgentRole.DIRECTOR,
        payload={"type": "patient_input", "patient_data": case},
    )
    response = await director.handle_message(message)
    result = response.payload

    if result.get("status") != "completed":
        print("执行失败:", result.get("error"))
        return result

    print(f"分诊等级: {result['triage_level']}")
    print(f"状态路径: {' -> '.join(result['profile']['state_history'])}")
    for note in result.get("retrieval_notes", []):
        print(f"检索说明: {note}")

    report = result.get("report") or {}
    if report:
        print(f"报告章节: {len(report.get('sections', []))} 节，"
              f"生成方式: {report.get('generated_by')}，"
              f"幻觉风险: {report.get('hallucination_risk')}")

    judge_result = result.get("judge_result")
    if judge_result:
        print(f"质检总分: {judge_result['overall_score']}，"
              f"是否通过: {'是' if judge_result['passed'] else '否'}")
    return result


CASE_1 = {
    "patient_id": "IBD-2026-001",
    "modality": "fusion",
    "clinical_table": {
        "age": 32,
        "disease_type": "CD",
        "disease_duration_months": 48,
        "current_biologic": "infliximab",
        "prior_biologics": ["adalimumab"],
        "immunomodulators": ["azathioprine"],
        "cdai": 280,
        "calprotectin": 450,
        "crp": 12.5,
        "candidate_genes": ["NOD2", "ATG16L1", "FCGR3A"],
    },
    "questionnaire": "腹痛、腹泻加重 2 周，每日排便 4-6 次，偶见黏液血便。",
}

CASE_2 = {
    "patient_id": "IBD-2026-002",
    "modality": "table",
    "clinical_table": {
        "age": 28,
        "disease_type": "UC",
        "disease_duration_months": 24,
        "current_biologic": "vedolizumab",
        "prior_biologics": [],
        "mayo_score": 2,
        "calprotectin": 120,
        "crp": 3.2,
    },
}


async def main() -> None:
    director = await build_system()
    try:
        results = {}
        for case in (CASE_1, CASE_2):
            results[case["patient_id"]] = await run_case(director, case)

        output_dir = PROJECT_ROOT / "outputs"
        output_dir.mkdir(exist_ok=True)
        out_path = output_dir / "demo_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n结果已写入 {out_path}")
        print("工具健康状态:", director.tools.health_check())
    finally:
        await director.shutdown()
        await director.tools.get("pubmed_search").aclose()
        await director.tools.get("llm_generate").aclose()


if __name__ == "__main__":
    asyncio.run(main())
