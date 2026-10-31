"""导演 Agent：接收患者输入，按复杂度分诊，用状态机驱动下游 Agent。

分诊规则：
- simple：只有临床表格、没有问卷文本 -> 分析 -> 报告
- moderate：表格 + 文本，或无影像的融合输入 -> 检索 -> 分析 -> 报告
- complex：带内镜/病理影像 -> 检索 -> 分析 -> 报告 -> 质检
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from models.schemas import (
    AgentMessage, AgentRole, PatientInput, TriageLevel,
)
from core.agent_base import AgentBase
from core.state_machine import State, StateConfig, StateMachine


class WaitingState(State):
    def __init__(self) -> None:
        super().__init__(StateConfig(name="WAITING", timeout_seconds=10,
                                    allowed_transitions=["TRIAGE"]))

    async def on_enter(self, context: Dict[str, Any]) -> None:
        return None

    async def on_execute(self, context: Dict[str, Any]) -> Optional[str]:
        return "TRIAGE" if context.get("patient_input") is not None else None


class TriageState(State):
    def __init__(self) -> None:
        super().__init__(StateConfig(
            name="TRIAGE", timeout_seconds=5,
            allowed_transitions=["RETRIEVAL", "ANALYSIS"],
        ))

    async def on_enter(self, context: Dict[str, Any]) -> None:
        return None

    async def on_execute(self, context: Dict[str, Any]) -> Optional[str]:
        patient: PatientInput = context["patient_input"]
        has_image = bool(patient.endoscopy_images or patient.pathology_images)
        has_text = bool(patient.questionnaire or patient.clinical_notes)

        if has_image:
            level, first_state = TriageLevel.COMPLEX, "RETRIEVAL"
        elif patient.modality.value == "table" and not has_text:
            level, first_state = TriageLevel.SIMPLE, "ANALYSIS"
        else:
            level, first_state = TriageLevel.MODERATE, "RETRIEVAL"

        paths = {
            TriageLevel.SIMPLE: ["ANALYSIS", "REPORT", "COMPLETED"],
            TriageLevel.MODERATE: ["RETRIEVAL", "ANALYSIS", "REPORT", "COMPLETED"],
            TriageLevel.COMPLEX: ["RETRIEVAL", "ANALYSIS", "REPORT", "JUDGE", "COMPLETED"],
        }
        context["triage_level"] = level
        context["execution_path"] = paths[level]
        return first_state


class InvokeState(State):
    """调用某个下游 Agent，保存结果，按 execution_path 决定下一站。"""

    def __init__(self, name: str, role: AgentRole, next_states: List[str],
                 fallback: Optional[str] = None):
        super().__init__(StateConfig(
            name=name, timeout_seconds=90, max_retries=1,
            fallback_state=fallback, allowed_transitions=next_states,
        ))
        self.role = role

    async def on_enter(self, context: Dict[str, Any]) -> None:
        return None

    async def on_execute(self, context: Dict[str, Any]) -> Optional[str]:
        director: DirectorAgent = context["director"]
        patient: PatientInput = context["patient_input"]

        payload: Dict[str, Any] = {"patient_input": patient.model_dump()}
        if self.name in ("ANALYSIS", "REPORT", "JUDGE"):
            payload["evidence_chain"] = context.get("evidence_chain")
        if self.name in ("REPORT", "JUDGE"):
            payload["analysis_result"] = context.get("analysis_result")
        if self.name in ("REPORT", "JUDGE"):
            payload["report"] = context.get("report")

        response = await director.send_to(
            self.role, payload, correlation_id=context.get("task_id")
        )
        data = response.payload
        if data.get("error"):
            raise RuntimeError(f"{self.name} 返回错误: {data['error']}")

        if self.name == "RETRIEVAL":
            context["evidence_chain"] = data.get("evidence_chain")
            context["retrieval_notes"] = data.get("evidence_chain", {}).get("notes", [])
        elif self.name == "ANALYSIS":
            context["analysis_result"] = data.get("analysis_result")
        elif self.name == "REPORT":
            context["report"] = data.get("report")
        elif self.name == "JUDGE":
            context["judge_result"] = data.get("judge_result")

        path: List[str] = context["execution_path"]
        idx = path.index(self.name)
        return path[idx + 1] if idx + 1 < len(path) else "COMPLETED"


class CompletedState(State):
    def __init__(self) -> None:
        super().__init__(StateConfig(name="COMPLETED", timeout_seconds=1,
                                    allowed_transitions=["__END__"]))

    async def on_enter(self, context: Dict[str, Any]) -> None:
        return None

    async def on_execute(self, context: Dict[str, Any]) -> Optional[str]:
        return "__END__"


class DirectorAgent(AgentBase):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(AgentRole.DIRECTOR, **kwargs)

    async def on_init(self) -> None:
        return None

    def _build_state_machine(self, task_id: str, patient: PatientInput) -> StateMachine:
        sm = StateMachine(name=f"task_{task_id}", initial_state="WAITING")
        sm.register_state(WaitingState())
        sm.register_state(TriageState())
        sm.register_state(InvokeState("RETRIEVAL", AgentRole.RETRIEVAL,
                                      ["ANALYSIS"], fallback="ANALYSIS"))
        sm.register_state(InvokeState("ANALYSIS", AgentRole.ANALYSIS, ["REPORT"]))
        sm.register_state(InvokeState("REPORT", AgentRole.REPORT,
                                      ["JUDGE", "COMPLETED"]))
        sm.register_state(InvokeState("JUDGE", AgentRole.JUDGE,
                                      ["COMPLETED"], fallback="COMPLETED"))
        sm.register_state(CompletedState())
        sm.set_context({
            "task_id": task_id,
            "patient_input": patient,
            "director": self,
        })
        return sm

    async def on_message(self, message: AgentMessage) -> Optional[Dict[str, Any]]:
        if message.payload.get("type") != "patient_input":
            return {"error": f"不支持的消息类型: {message.payload.get('type')}"}

        task_id = message.correlation_id
        patient = PatientInput(**message.payload.get("patient_data", {}))
        sm = self._build_state_machine(task_id, patient)

        try:
            final_ctx = await sm.run()
        except Exception as exc:
            return {"task_id": task_id, "status": "failed", "error": str(exc),
                    "profile": sm.profile()}

        return {
            "task_id": task_id,
            "status": "completed",
            "triage_level": final_ctx.get("triage_level").value,
            "evidence_chain": final_ctx.get("evidence_chain"),
            "analysis_result": final_ctx.get("analysis_result"),
            "report": final_ctx.get("report"),
            "judge_result": final_ctx.get("judge_result"),
            "retrieval_notes": final_ctx.get("retrieval_notes", []),
            "errors": final_ctx.get("errors", []),
            "profile": sm.profile(),
        }

    async def on_shutdown(self) -> None:
        return None
