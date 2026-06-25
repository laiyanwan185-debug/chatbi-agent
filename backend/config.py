"""
配置管理中心 — Pydantic BaseSettings (championship-grade)
"""
from __future__ import annotations

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM API (使用 SecretStr 彻底防泄漏保护) ──
    ANTHROPIC_API_KEY: SecretStr = SecretStr("")
    OPENAI_API_KEY: SecretStr = SecretStr("")
    DEEPSEEK_API_KEY: SecretStr = SecretStr("")

    #  高度灵活的 Base URL 配置，完美适配国内中转与官方域名
    ANTHROPIC_BASE_URL: str = "https://api.anthropic.com"
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"

    ANTHROPIC_MODEL: str = "claude-3-5-sonnet-20240620"  # 推荐使用稳定的 3.5 Sonnet
    OPENAI_MODEL: str = "gpt-4o"
    DEEPSEEK_MODEL: str = "deepseek-chat"

    LLM_MAX_RETRIES: int = 5
    LLM_TIMEOUT: int = 60

    # ── 配置文件路径 ──
    INDICATORS_CONFIG_PATH: str = "app/configs/indicators.yaml"
    JOIN_GRAPH_CONFIG_PATH: str = "app/configs/join_graph.yaml"

    # ── 数据库  ──
    DB_DSN: str = "postgresql://user:pass@localhost:5432/chatbi"
    DB_POOL_MIN: int = 2
    DB_POOL_MAX: int = 10
    DB_CONNECT_TIMEOUT: int = 10
    DB_COMMAND_TIMEOUT: int = 30  #  统一修改为：秒(seconds)，完美对接 asyncpg.Pool 物理约束

    # ── Embedding ──
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_BATCH_SIZE: int = 32
    SCHEMA_TOP_K: int = 5

    # ── 语义缓存 ──
    CACHE_BACKEND: str = "diskcache"
    CACHE_SIMILARITY_THRESHOLD: float = 0.98
    CACHE_MAX_ENTRIES: int = 1000
    REDIS_DSN: str = "redis://localhost:6379/0"

    # ── 反馈闸门 ──
    CONFIDENCE_THRESHOLD: float = 0.85
    EVAL_PASS_THRESHOLD: float = 0.85
    MAX_AGENT_STEPS: int = 3
    EVAL_WEIGHTS: dict[str, float] = {
        "correctness": 0.30,
        "completeness": 0.20,
        "consistency": 0.20,
        "interpretability": 0.15,
        "display_fitness": 0.15,
    }

    # ── DAG ──
    DAG_GLOBAL_TIMEOUT: int = 60
    DAG_SQL_TIMEOUT: int = 30
    DAG_PYTHON_TIMEOUT: int = 15
    DAG_MERGE_TIMEOUT: int = 5
    DAG_THREAD_POOL_SIZE: int = 8

    # ── 执行器 ──
    SANDBOX_MEMORY_LIMIT_MB: int = 1024
    SANDBOX_PYTHON_TIMEOUT: int = 15
    PROBE_TIMEOUT: int = 5
    PROBE_MAX_ROUNDS: int = 1
    DATA_WARNING_FLAG: str = "Data_Warning"

    # ── 前端 ──
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:3001"]

    # ── 日志 ──
    LOG_LEVEL: str = "INFO"
    TRACE_ENABLED: bool = True

    # ── HuggingFace 镜像（国内加速）──
    HF_ENDPOINT: str = ""


settings = Settings()