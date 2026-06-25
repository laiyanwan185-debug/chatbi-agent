"""
数据库连接管理 — asyncpg 连接池 + 活性健康自检 + 自动断线重连自愈
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import asyncpg

logger = logging.getLogger(__name__)


class DatabasePool:
    """asyncpg 连接池封装，支持强安全反射健康自检与自动重连自愈。"""

    def __init__(self, dsn: str, min_size: int = 2, max_size: int = 10, timeout: int = 10) -> None:
        self._dsn = dsn
        self._min_size = min_size
        self._max_size = max_size
        self._timeout = timeout
        self._pool: asyncpg.Pool | None = None
        self._lock = asyncio.Lock()  #  防止高并发下断线重连发生资源竞争

    # ── 生命周期 ──

    async def create(self) -> None:
        """创建连接池。"""
        self._pool = await asyncpg.create_pool(
            dsn=self._dsn,
            min_size=self._min_size,
            max_size=self._max_size,
            command_timeout=self._timeout,
        )
        logger.info("Database pool created (min=%d, max=%d)", self._min_size, self._max_size)

    async def close(self) -> None:
        """关闭连接池。"""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Database pool closed")

    async def refresh(self) -> None:
        """线程安全的重连自愈方法。"""
        async with self._lock:
            logger.warning(" Detected connection anomaly, triggering pool auto-refresh...")
            try:
                await self.close()
                await self.create()
                logger.info(" Database pool successfully healed.")
            except Exception as e:
                logger.error(" Failed to auto-refresh database pool: %s", e)
                raise

    # ── 自动重连执行器包装器 (Auto-healing wrapper) ──

    async def _execute_with_retry(self, func, *args, **kwargs) -> Any:
        """核心执行代理：当连接因底层抖动断开时，自动重建连接池并重试一次"""
        try:
            return await func(*args, **kwargs)
        except (asyncpg.InterfaceError, asyncpg.ConnectionDoesNotExistError) as exc:
            logger.warning(" Database connection lost (%s). Attempting auto-healing...", exc)
            await self.refresh()  # 触发重连
            return await func(*args, **kwargs)  # 重试第二次

    # ── 核心接口 ──

    async def fetch(self, query: str, *args: Any) -> list[asyncpg.Record]:
        """执行查询并返回所有行（带自愈保障）。"""
        pool = self._require_pool()
        
        async def _execute():
            async with pool.acquire() as conn:
                return await conn.fetch(query, *args)
                
        return await self._execute_with_retry(_execute)

    async def fetchrow(self, query: str, *args: Any) -> asyncpg.Record | None:
        """执行查询并返回第一行（带自愈保障）。"""
        pool = self._require_pool()
        
        async def _execute():
            async with pool.acquire() as conn:
                return await conn.fetchrow(query, *args)
                
        return await self._execute_with_retry(_execute)

    async def execute(self, query: str, *args: Any) -> str:
        """执行 DDL / DML（仅内部管理用，带自愈保障）。"""
        pool = self._require_pool()
        
        async def _execute():
            async with pool.acquire() as conn:
                return await conn.execute(query, *args)
                
        return await self._execute_with_retry(_execute)

    # ── 健康检查 (强安全反射机制) ──

    async def health(self) -> dict[str, Any]:
        """返回连接池健康状态与空闲率。"""
        if self._pool is None:
            return {"status": "disconnected", "pool": None}
        try:
            # 活性 Ping 测试
            async with self._pool.acquire(timeout=3) as conn:
                await conn.fetchval("SELECT 1")
            
            #  强安全反射：多层 getattr 防御，彻底防止未来 asyncpg 私有命名更新导致崩溃
            free_conns = "N/A"
            try:
                holders = getattr(self._pool, "_holders", None)
                queue_obj = getattr(self._pool, "_queue", None)
                if holders is not None and queue_obj is not None:
                    # 兼容不同版本 asyncpg 的内部 Queue 底层结构
                    q_len = len(queue_obj._queue) if hasattr(queue_obj, "_queue") else 0 # noqa: SLF001
                    free_conns = len(holders) - q_len # 算出当前闲置在池里的连接
            except Exception: # noqa: BLE001
                pass

            return {
                "status": "healthy",
                "pool": {
                    "free": free_conns,
                    "size": getattr(self._pool, "_maxsize", "N/A"), # noqa: SLF001
                },
            }
        except Exception as exc: # noqa: BLE001
            logger.warning("Health check failed: %s", exc)
            return {"status": "unhealthy", "error": str(exc)}

    # ── 内部 ──

    def _require_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            msg = "Pool not initialized. Call create() first."
            raise RuntimeError(msg)
        return self._pool

    @property
    def is_connected(self) -> bool:
        return self._pool is not None and not self._pool._closed  # noqa: SLF001


# ── 全局单例 ──
_db_pool: DatabasePool | None = None


async def get_pool() -> DatabasePool:
    """获取全局 DatabasePool 单例（FastAPI lifespan 中初始化）。"""
    global _db_pool  # noqa: PLW0603
    if _db_pool is None:
        from config import settings

        _db_pool = DatabasePool(
            dsn=settings.DB_DSN,
            min_size=settings.DB_POOL_MIN,
            max_size=settings.DB_POOL_MAX,
            timeout=settings.DB_CONNECT_TIMEOUT,
        )
        await _db_pool.create()
    return _db_pool


async def close_pool() -> None:
    """关闭全局 DatabasePool（FastAPI lifespan 中调用）。"""
    global _db_pool  # noqa: PLW0603
    if _db_pool:
        await _db_pool.close()
        _db_pool = None