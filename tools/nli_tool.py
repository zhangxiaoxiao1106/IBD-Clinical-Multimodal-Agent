"""NLI 校验工具：判断证据(premise)与结论(hypothesis)是蕴含/矛盾/中立。

模型（roberta-large-mnli）首次调用时才加载；torch/transformers 缺失
或模型加载失败时，工具层降级为 neutral，由上层在报告中说明。
"""

from __future__ import annotations

from typing import Any, Dict

from core.tool_registry import BaseTool, ToolSchema
from tools.base import NLIValidateInput, NLIValidateOutput


class NLITool(BaseTool):
    # roberta-large-mnli 的三个 logit 顺序
    LABELS = {0: "contradiction", 1: "neutral", 2: "entailment"}

    def __init__(self, model_name: str = "roberta-large-mnli"):
        super().__init__(ToolSchema(
            name="nli_validate",
            description="NLI 事实一致性校验",
            input_schema=NLIValidateInput,
            output_schema=NLIValidateOutput,
            timeout_seconds=30.0,
            max_retries=0,
            fallback_value={"label": "neutral", "confidence": 0.33, "available": False},
        ))
        self.model_name = model_name
        self._tokenizer = None
        self._model = None
        self._device = "cpu"

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(f"缺少 transformers/torch 依赖: {exc}")

        if torch.cuda.is_available():
            self._device = "cuda"
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.to(self._device).eval()

    async def _execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        self._load()  # 依赖缺失时抛异常，由工具层降级
        import torch
        from torch.nn.functional import softmax

        params = NLIValidateInput(**inputs)
        encoded = self._tokenizer(
            params.premise, params.hypothesis,
            return_tensors="pt", truncation=True, max_length=512,
        )
        with torch.no_grad():
            logits = self._model(
                input_ids=encoded["input_ids"].to(self._device),
                attention_mask=encoded["attention_mask"].to(self._device),
            ).logits
            probs = softmax(logits, dim=1)[0]

        idx = int(torch.argmax(probs))
        return {
            "label": self.LABELS[idx],
            "confidence": float(probs[idx]),
            "available": True,
        }
