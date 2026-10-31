"""一个简单的异步状态机：注册状态 -> 按 on_execute 返回的状态名转移。

支持：
- 合法转移校验（allowed_transitions）
- 单状态超时（wait_for）
- 执行失败重试（max_retries），重试次数用完后交给状态自己决定降级目标
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from core.logging_setup import get_logger

logger = get_logger("state_machine")


class StateTransitionError(RuntimeError):
    pass


class StateTimeoutError(RuntimeError):
    pass


@dataclass
class StateConfig:
    name: str
    timeout_seconds: float = 60.0
    max_retries: int = 0
    fallback_state: Optional[str] = None
    allowed_transitions: List[str] = field(default_factory=list)


class State(ABC):
    def __init__(self, config: StateConfig):
        self.config = config
        self.entered_at: Optional[float] = None

    @property
    def name(self) -> str:
        return self.config.name

    async def enter(self, context: Dict[str, Any]) -> None:
        self.entered_at = time.time()
        await self.on_enter(context)

    async def execute(self, context: Dict[str, Any]) -> Optional[str]:
        """执行状态，返回下一状态名；返回 None 表示原地重试/等待。"""
        for attempt in range(self.config.max_retries + 1):
            start = time.time()
            try:
                next_state = await asyncio.wait_for(
                    self.on_execute(context), timeout=self.config.timeout_seconds
                )
                logger.info("状态 %s 执行完成，用时 %.3fs", self.name, time.time() - start)
                return next_state
            except asyncio.TimeoutError:
                logger.warning("状态 %s 超时（第 %d 次）", self.name, attempt + 1)
                if attempt >= self.config.max_retries:
                    if self.config.fallback_state:
                        return self.config.fallback_state
                    raise StateTimeoutError(f"状态 {self.name} 超时")
            except Exception as exc:
                logger.warning("状态 %s 执行失败（第 %d 次）: %s", self.name, attempt + 1, exc)
                if attempt >= self.config.max_retries:
                    if self.config.fallback_state:
                        context.setdefault("errors", []).append(
                            f"{self.name}: {exc}，降级到 {self.config.fallback_state}"
                        )
                        return self.config.fallback_state
                    raise StateTransitionError(f"状态 {self.name} 失败: {exc}") from exc
                await asyncio.sleep(0.5 * (attempt + 1))
        return None

    @abstractmethod
    async def on_enter(self, context: Dict[str, Any]) -> None: ...

    @abstractmethod
    async def on_execute(self, context: Dict[str, Any]) -> Optional[str]: ...

    async def on_exit(self, context: Dict[str, Any]) -> None:
        pass


class StateMachine:
    def __init__(self, name: str, initial_state: str):
        self.name = name
        self.initial_state = initial_state
        self._states: Dict[str, State] = {}
        self._transitions: Dict[str, Set[str]] = {}
        self._current: Optional[str] = None
        self._context: Dict[str, Any] = {}
        self._history: List[str] = []
        self._durations: Dict[str, float] = {}

    def register_state(self, state: State) -> None:
        self._states[state.name] = state
        self._transitions[state.name] = set(state.config.allowed_transitions)

    def set_context(self, context: Dict[str, Any]) -> None:
        self._context.update(context)

    @property
    def context(self) -> Dict[str, Any]:
        return self._context

    def current_state(self) -> Optional[str]:
        return self._current

    async def step(self) -> bool:
        if self._current is None:
            await self._transition_to(self.initial_state)
            return True

        state = self._states[self._current]
        start = time.time()
        next_name = await state.execute(self._context)
        self._durations[state.name] = self._durations.get(state.name, 0.0) + time.time() - start

        if next_name is None:
            return True
        if next_name == "__END__":
            await state.on_exit(self._context)
            self._current = None
            return False

        allowed = self._transitions.get(state.name, set())
        if next_name not in allowed:
            raise StateTransitionError(f"非法转移 {state.name} -> {next_name}，允许: {sorted(allowed)}")

        await state.on_exit(self._context)
        await self._transition_to(next_name)
        return True

    async def run(self) -> Dict[str, Any]:
        while await self.step():
            pass
        return self._context

    def profile(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "state_history": list(self._history),
            "state_durations": {k: round(v, 3) for k, v in self._durations.items()},
            "total_duration": round(sum(self._durations.values()), 3),
        }

    async def _transition_to(self, name: str) -> None:
        if name not in self._states:
            raise StateTransitionError(f"状态 {name} 未注册")
        self._current = name
        self._history.append(name)
        logger.info("转移到 %s", name)
        await self._states[name].enter(self._context)
