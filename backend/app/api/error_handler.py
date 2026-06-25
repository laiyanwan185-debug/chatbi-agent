"""全局异常处理中间件。

统一所有错误响应为结构化格式：
  {"detail": "人类可读信息", "trace_id": "..."}

用法：
  from app.api.error_handler import register_error_handlers
  register_error_handlers(app)
"""

from __future__ import annotations

import logging
import traceback as tb
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


def _trace_id() -> str:
    """生成简短 trace_id 用于错误关联。"""
    return uuid.uuid4().hex[:12]


def register_error_handlers(app: FastAPI) -> None:
    """注册所有全局异常处理器到 FastAPI 应用。"""

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        tid = _trace_id()
        errors = []
        for err in exc.errors():
            loc = ".".join(str(p) for p in err.get("loc", []))
            msg = err.get("msg", "")
            errors.append(f"{loc}: {msg}")
        detail = "请求参数校验失败: " + "; ".join(errors)
        logger.warning("ValidationError trace=%s: %s", tid, detail)
        return JSONResponse(
            status_code=422,
            content={"detail": detail, "trace_id": tid},
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        tid = _trace_id()
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "trace_id": tid},
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        tid = _trace_id()
        logger.error("Unhandled exception trace=%s\n%s", tid, tb.format_exc())
        return JSONResponse(
            status_code=500,
            content={
                "detail": "系统内部错误，请稍后重试",
                "trace_id": tid,
            },
        )
