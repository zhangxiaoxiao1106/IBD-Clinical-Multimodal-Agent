"""日志与 trace_id。用标准库 logging，trace_id 放在日志的固定字段里。"""

from __future__ import annotations

import logging
import uuid
from contextvars import ContextVar
from typing import Any, Optional

_current_trace: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)


class _TraceFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = _current_trace.get() or "-"
        return True


def setup_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.addFilter(_TraceFilter())
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s [%(trace_id)s] %(message)s")
    )
    root = logging.getLogger("ibd")
    root.handlers[:] = [handler]
    root.setLevel(level)
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"ibd.{name}")


def new_trace_id() -> str:
    return uuid.uuid4().hex[:12]


class Trace:
    """一次调用的 trace 上下文，with 语法把 trace_id 放进 ContextVar。"""

    def __init__(self, operation: str, trace_id: Optional[str] = None):
        self.operation = operation
        self.trace_id = trace_id or new_trace_id()
        self._token: Any = None
        self._logger = get_logger("trace")

    def __enter__(self) -> "Trace":
        self._token = _current_trace.set(self.trace_id)
        self._logger.info("%s 开始", self.operation)
        return self

    def __exit__(self, *exc: Any) -> None:
        if exc[0] is not None:
            self._logger.warning("%s 异常: %s", self.operation, exc[1])
        _current_trace.reset(self._token)

    def event(self, message: str, **fields: Any) -> None:
        if fields:
            self._logger.info("%s: %s %s", self.operation, message, fields)
        else:
            self._logger.info("%s: %s", self.operation, message)
