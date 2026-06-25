"""
统一大模型客户端引擎 — 多模型策略模式 + 协程安全 Token 计数 + 声明式指数退避重试
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from contextvars import ContextVar
from typing import Any, Dict, List

import anthropic
import openai
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import settings

logger = logging.getLogger(__name__)

#  协程安全的 Token 计数器（ContextVar 保证并发请求之间的数据绝对隔离与安全）
# 结构: {"input_tokens": 0, "output_tokens": 0, "model": ""}
token_usage_var: ContextVar[Dict[str, Any]] = ContextVar(
    "token_usage", 
    default={"input_tokens": 0, "output_tokens": 0, "model": ""}
)


# ── 1. 统一客户端抽象基类 (Strategy Pattern) ──
class BaseLLMClient(ABC):
    """大模型接口契约，所有具体模型策略必须继承并实现此类。"""

    @abstractmethod
    async def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """
        统一的异步对话接口。
        
        Args:
            messages: 统一的格式化消息列表，格式为 [{"role": "system"|"user"|"assistant", "content": "..."}]
            **kwargs: 覆盖默认温度(temperature)等模型参数
            
        Returns:
            str: 大模型生成的干净文本内容
        """
        pass


# ── 2. OpenAI / DeepSeek 策略实现 ──
class OpenAICompatibleClient(BaseLLMClient):
    """OpenAI 规范客户端（完美兼容 DeepSeek / Ollama / 零一万物等）"""

    def __init__(self, api_key: str, base_url: str, default_model: str) -> None:
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=settings.LLM_TIMEOUT,
        )
        self._default_model = default_model

    # 声明式重试：仅在遇到网络超时、连接错误或被限流(429)时，触发指数退避重试
    @retry(
        reraise=True,
        stop=stop_after_attempt(settings.LLM_MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.RateLimitError
        )),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        model = kwargs.pop("model", self._default_model)
        temp = kwargs.pop("temperature", 0.0)  # 默认 0.0，保证 SQL 生成的高确定性

        logger.debug("Requesting OpenAI-compatible model: %s", model)
        response = await self._client.chat.completions.create(
            model=model,
            messages=messages, # type: ignore
            temperature=temp,
            **kwargs
        )

        #  协程安全 Token 计数提取与累加
        usage = response.usage
        if usage:
            current_usage = token_usage_var.get().copy()
            current_usage["input_tokens"] += usage.prompt_tokens
            current_usage["output_tokens"] += usage.completion_tokens
            current_usage["model"] = model
            token_usage_var.set(current_usage)
            logger.info(
                "Token Usage [%s] -> Input: %d, Output: %d", 
                model, usage.prompt_tokens, usage.completion_tokens
            )

        return response.choices[0].message.content or ""


# ── 3. Anthropic (Claude 3.5) 策略实现 ──
class AnthropicClient(BaseLLMClient):
    """Claude 3.5 专用原生异步客户端"""

    def __init__(self, api_key: str, base_url: str, default_model: str) -> None:
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key,
            base_url=base_url,
            timeout=settings.LLM_TIMEOUT,
        )
        self._default_model = default_model

    @retry(
        reraise=True,
        stop=stop_after_attempt(settings.LLM_MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((
            anthropic.APIConnectionError,
            anthropic.APITimeoutError,
            anthropic.RateLimitError
        )),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        model = kwargs.pop("model", self._default_model)
        temp = kwargs.pop("temperature", 0.0)
        max_tokens = kwargs.pop("max_tokens", 4096)

        #  避坑防线：剥离 system 提示词，转换为 Anthropic 原生顶层参数
        system_prompt: str | None = None
        filtered_messages: List[Dict[str, str]] = []

        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                filtered_messages.append(msg)

        logger.debug("Requesting Anthropic (Claude) model: %s", model)

        # 组装请求参数
        response = await self._client.messages.create(
            model=self.model_name_resolve(model),
            messages=filtered_messages,
            max_tokens=max_tokens,
            temperature=temp,
            **({"system": system_prompt} if system_prompt else {}),
        )

        #  协程安全 Token 计数提取
        usage = response.usage
        if usage:
            current_usage = token_usage_var.get().copy()
            current_usage["input_tokens"] += usage.input_tokens
            current_usage["output_tokens"] += usage.output_tokens
            current_usage["model"] = model
            token_usage_var.set(current_usage)
            logger.info(
                "Token Usage [Claude:%s] -> Input: %d, Output: %d", 
                model, usage.input_tokens, usage.output_tokens
            )

        # 统一将内容解析为纯文本
        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text
        return content

    @staticmethod
    def model_name_resolve(config_model: str) -> str:
        """确保在 env 中配置的简称能正确映射为 Anthropic 官方物理名称"""
        if "claude" not in config_model.lower():
            return "claude-3-5-sonnet-20240620"
        return config_model


# ── 4. 动态工厂注册中心 ──

class LLMFactory:
    _registry: Dict[str, BaseLLMClient] = {}

    @classmethod
    def get_client(cls, provider: str) -> BaseLLMClient:
        """
        根据供应商名称（openai, deepseek, anthropic）获取单例客户端
        """
        provider = provider.lower().strip()
        if provider not in cls._registry:
            cls._registry[provider] = cls._build_client(provider)
        return cls._registry[provider]

    @classmethod
    def _build_client(cls, provider: str) -> BaseLLMClient:
        # 统一读取 Settings 中的 API Key (使用 get_secret_value() 保证安全)
        if provider == "openai":
            return OpenAICompatibleClient(
                api_key=settings.OPENAI_API_KEY.get_secret_value(),
                base_url=settings.OPENAI_BASE_URL,
                default_model=settings.OPENAI_MODEL,
            )
        elif provider == "deepseek":
            return OpenAICompatibleClient(
                api_key=settings.DEEPSEEK_API_KEY.get_secret_value(),
                base_url=settings.DEEPSEEK_BASE_URL,
                default_model=settings.DEEPSEEK_MODEL,
            )
        elif provider == "anthropic":
            return AnthropicClient(
                api_key=settings.ANTHROPIC_API_KEY.get_secret_value(),
                base_url=settings.ANTHROPIC_BASE_URL,
                default_model=settings.ANTHROPIC_MODEL,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")


# ── 5. 全局简易快捷调用接口 ──

async def chat_with_model(
    provider: str, 
    messages: List[Dict[str, str]], 
    **kwargs: Any
) -> str:
    """全局快捷异步调用通道，支持 Token 自动追踪。"""
    client = LLMFactory.get_client(provider)
    return await client.chat(messages, **kwargs)