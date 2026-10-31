"""分诊与状态机路径的测试。"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.schemas import PatientInput, TriageLevel
from agents.director import TriageState


def run(coro):
    return asyncio.run(coro)


def triage(case: dict):
    state = TriageState()
    context = {"patient_input": PatientInput(**case)}
    next_state = run(state.on_execute(context))
    return context["triage_level"], next_state, context["execution_path"]


def test_table_only_is_simple():
    level, next_state, path = triage({
        "patient_id": "p1", "modality": "table",
        "clinical_table": {"disease_type": "UC"},
    })
    assert level == TriageLevel.SIMPLE
    assert next_state == "ANALYSIS"
    assert "RETRIEVAL" not in path


def test_table_with_questionnaire_is_moderate():
    level, next_state, path = triage({
        "patient_id": "p2", "modality": "fusion",
        "clinical_table": {"disease_type": "CD"},
        "questionnaire": "腹泻两周",
    })
    assert level == TriageLevel.MODERATE
    assert next_state == "RETRIEVAL"
    assert "JUDGE" not in path


def test_image_is_complex():
    level, next_state, path = triage({
        "patient_id": "p3", "modality": "image",
        "endoscopy_images": ["a.jpg"],
    })
    assert level == TriageLevel.COMPLEX
    assert "JUDGE" in path
    assert next_state == "RETRIEVAL"
