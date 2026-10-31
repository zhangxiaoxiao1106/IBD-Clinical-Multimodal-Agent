"""OpenAI 兼容接口的 LLM 调用（vLLM / TGI / OpenAI 均可）。

服务连不上时直接抛异常，由工具层走 fallback，
报告 Agent 检测到 fallback 后改用本地模板生成，不会把错误提示写进报告。
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from core.tool_registry import BaseTool, ToolSchema
from tools.base import LLMGenerateInput, LLMGenerateOutput

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore


class LLMTool(BaseTool):
    def __init__(self, api_base: str = "http://localhost:8000/v1",
                 model: str = "default", api_key: Optional[str] = None):
        super().__init__(ToolSchema(
            name="llm_generate",
            description="OpenAI 兼容接口的文本生成",
            input_schema=LLMGenerateInput,
            output_schema=LLMGenerateOutput,
            timeout_seconds=45.0,
            max_retries=1,
            fallback_value={"generated_text": "", "finish_reason": "unavailable",
                            "total_tokens": 0, "generation_time_ms": 0.0},
        ))
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.api_key = api_key
        self._client = httpx.AsyncClient(timeout=45.0) if httpx is not None else None

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()

    async def _execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self._client is None:
            raise RuntimeError("httpx 未安装")

        params = LLMGenerateInput(**inputs)
        start = time.time()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": params.system_prompt or "你是一个严谨的医学报告助手。"},
                {"role": "user", "content": params.prompt},
            ],
            "temperature": params.temperature,
            "max_tokens": params.max_tokens,
        }
        if "json" in params.prompt.lower():
            payload["response_format"] = {"type": "json_object"}

        resp = await self._client.post(
            f"{self.api_base}/chat/completions", json=payload, headers=headers
        )
        resp.raise_for_status()
        data = resp.json()
        choice = data.get("choices", [{}])[0]
        usage = data.get("usage", {})

        return {
            "generated_text": choice.get("message", {}).get("content", ""),
            "finish_reason": choice.get("finish_reason", "stop"),
            "total_tokens": usage.get("total_tokens", 0),
            "generation_time_ms": (time.time() - start) * 1000,
        }
