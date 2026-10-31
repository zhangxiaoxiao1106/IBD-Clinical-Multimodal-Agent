"""统一工具基类与注册中心。

每个外部能力（PubMed、向量库、LLM、NLI）封装成一个 Tool，
这里统一做输入校验、超时、重试和失败降级。
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, ValidationError

from core.logging_setup import get_logger

logger = get_logger("tool")


class ToolError(RuntimeError):
    pass


@dataclass
class ToolSchema:
    name: str
    description: str
    input_schema: Type[BaseModel]
    output_schema: Optional[Type[BaseModel]] = None
    timeout_seconds: float = 30.0
    max_retries: int = 1
    fallback_value: Optional[Any] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class ToolCall:
    tool_name: str
    inputs: Dict[str, Any]
    outputs: Optional[Any] = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    degraded: bool = False


class BaseTool(ABC):
    def __init__(self, schema: ToolSchema):
        self.schema = schema
        self.history: List[ToolCall] = []
        self.health: str = "healthy"

    @abstractmethod
    async def _execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]: ...

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        record = ToolCall(tool_name=self.schema.name, inputs=inputs)
        start = time.time()
        result: Any = None
        last_error: Optional[Exception] = None

        try:
            validated = self.schema.input_schema(**inputs)
            inputs = validated.model_dump()
        except ValidationError as exc:
            record.error = f"输入校验失败: {exc}"
            record.duration_ms = (time.time() - start) * 1000
            self.history.append(record)
            raise ToolError(record.error) from exc

        for attempt in range(self.schema.max_retries + 1):
            try:
                result = await asyncio.wait_for(
                    self._execute(inputs), timeout=self.schema.timeout_seconds
                )
                if self.schema.output_schema is not None:
                    result = self.schema.output_schema(**result).model_dump()
                last_error = None
                break
            except Exception as exc:  # noqa: BLE001 - 降级逻辑需要兜住所有异常
                last_error = exc
                logger.warning(
                    "工具 %s 第 %d 次执行失败: %s",
                    self.schema.name, attempt + 1, exc,
                )
                if attempt < self.schema.max_retries:
                    await asyncio.sleep(0.5 * (attempt + 1))

        if last_error is not None:
            record.duration_ms = (time.time() - start) * 1000
            if self.schema.fallback_value is not None:
                logger.warning("工具 %s 降级，返回 fallback_value", self.schema.name)
                result = self.schema.fallback_value
                record.degraded = True
                record.error = str(last_error)
                self.health = "degraded"
            else:
                record.error = str(last_error)
                self.history.append(record)
                raise ToolError(f"工具 {self.schema.name} 执行失败: {last_error}") from last_error

        record.outputs = result
        record.duration_ms = (time.time() - start) * 1000
        self.history.append(record)
        return result

    def recent_calls(self, limit: int = 10) -> List[ToolCall]:
        return self.history[-limit:]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        if tool.schema.name in self._tools:
            raise ValueError(f"工具 {tool.schema.name} 已注册")
        self._tools[tool.schema.name] = tool

    def get(self, name: str) -> BaseTool:
        if name not in self._tools:
            raise KeyError(f"工具 {name} 未注册")
        return self._tools[name]

    def has(self, name: str) -> bool:
        return name in self._tools

    async def call(self, name: str, inputs: Dict[str, Any]) -> Any:
        return await self.get(name).execute(inputs)

    def health_check(self) -> Dict[str, str]:
        return {name: tool.health for name, tool in self._tools.items()}
