"""Agent 基类：统一生命周期、消息收发、以及下游 Agent 的直连路由。

Agent 之间不经过消息队列，直接用 async 方法调用，调用链清晰、便于调试。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from models.schemas import AgentMessage, AgentRole, MessageType
from core.logging_setup import Trace, get_logger
from core.memory import MemoryManager
from core.tool_registry import ToolRegistry

logger = get_logger("agent")


class AgentBase(ABC):
    def __init__(
        self,
        role: AgentRole,
        memory: Optional[MemoryManager] = None,
        tools: Optional[ToolRegistry] = None,
    ):
        self.role = role
        self.memory = memory or MemoryManager()
        self.tools = tools or ToolRegistry()
        self.peers: Dict[AgentRole, "AgentBase"] = {}
        self._initialized = False
        self.message_count = 0
        self.error_count = 0

    @property
    def name(self) -> str:
        return self.role.value

    def register_peer(self, agent: "AgentBase") -> None:
        self.peers[agent.role] = agent

    async def initialize(self) -> None:
        if self._initialized:
            return
        await self.on_init()
        self._initialized = True

    async def shutdown(self) -> None:
        await self.on_shutdown()

    async def handle_message(self, message: AgentMessage) -> AgentMessage:
        self.message_count += 1
        with Trace(f"{self.name}.handle", trace_id=message.metadata.get("trace_id")) as trace:
            self.memory.short_term.add(
                key=f"msg_{message.message_id}",
                value={"type": message.message_type.value, "sender": message.sender.value},
                importance=0.8,
            )
            trace.event("收到消息", sender=message.sender.value, keys=list(message.payload.keys()))
            try:
                payload = await self.on_message(message)
            except Exception as exc:  # 业务异常包成错误消息，避免整条链路崩掉
                self.error_count += 1
                logger.exception("%s 处理消息失败", self.name)
                payload = {"error": str(exc), "agent": self.name}

            return AgentMessage(
                correlation_id=message.correlation_id,
                message_type=MessageType.RESPONSE,
                sender=self.role,
                receiver=message.sender,
                payload=payload or {},
                trace=message.trace + [self.name],
            )

    async def send_to(
        self,
        role: AgentRole,
        payload: Dict[str, Any],
        correlation_id: Optional[str] = None,
    ) -> AgentMessage:
        """直接调用下游 Agent 并等待返回。"""
        peer = self.peers.get(role)
        if peer is None:
            raise RuntimeError(f"下游 {role.value} 未注册，无法调用")
        message = AgentMessage(
            correlation_id=correlation_id or f"corr_{self.message_count}",
            message_type=MessageType.REQUEST,
            sender=self.role,
            receiver=role,
            payload=payload,
        )
        logger.info("%s -> %s", self.name, role.value)
        return await peer.handle_message(message)

    async def call_tool(self, tool_name: str, inputs: Dict[str, Any]) -> Any:
        return await self.tools.call(tool_name, inputs)

    def health(self) -> Dict[str, Any]:
        return {
            "role": self.name,
            "initialized": self._initialized,
            "messages": self.message_count,
            "errors": self.error_count,
            "memory_entries": len(self.memory.short_term),
        }

    @abstractmethod
    async def on_init(self) -> None: ...

    @abstractmethod
    async def on_message(self, message: AgentMessage) -> Optional[Dict[str, Any]]: ...

    async def on_shutdown(self) -> None:
        pass
