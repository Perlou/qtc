"""
Phase 10 辅助模块：LLM 客户端（带调用预算、本地缓存与降级）
============================================================

设计原则（对应 CONCEPT.md 第 11.4 节"工程与运营风险"）：
    1. 默认无密钥：所有课程在没有 API Key 时都要能完整跑通 → 调用失败返回 None，由课程走降级路径
    2. 可复现：temperature=0；每次请求的输入与输出按哈希缓存到 data/cache/phase10/llm/，
       相同输入第二次运行直接读缓存（缓存本身就是"数据的一部分"，应该版本化）
    3. 成本可控：每个进程最多 MAX_CALLS 次真实请求（默认 10）
    4. 离线（QTC_OFFLINE=1）：只读缓存，不发网络请求

支持的服务（按优先级）：
    OPENAI_API_KEY      openai SDK；可用 OPENAI_BASE_URL 指向任何 OpenAI 兼容服务（如 DeepSeek、Qwen）
    ANTHROPIC_API_KEY   需要另行安装官方 anthropic SDK（本课程环境未安装 → 自动降级）
    模型名用 LLM_MODEL 环境变量覆盖；默认见 DEFAULT_MODELS
"""

from __future__ import annotations

import hashlib
import json
import os
import re

from ai_tools import P10_CACHE
from utils.data import is_offline

DEFAULT_MODELS = {"openai": "gpt-4o-mini", "anthropic": "claude-sonnet-5"}
MAX_CALLS = 10
CACHE = P10_CACHE / "llm"


def parse_json(text: str):
    """从模型输出中解析第一个 JSON 对象或数组（兼容 ```json 代码块与前后多余文字）"""
    if text is None:
        return None
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "[{":
            try:
                obj, _ = dec.raw_decode(text[i:])
                return obj
            except json.JSONDecodeError:
                continue
    return None


class LLMClient:
    """极简 LLM 客户端：json(prompt) → dict / list / None"""

    calls_made = 0  # 进程级计数（所有实例共享）

    def __init__(self, max_calls: int = MAX_CALLS):
        self.max_calls = max_calls
        self.provider, self.model, self.reason = self._detect()
        self.cache_hits = 0
        self.errors: list[str] = []

    @staticmethod
    def _detect():
        if os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        elif os.getenv("ANTHROPIC_API_KEY"):
            try:
                import anthropic  # noqa: F401
                provider = "anthropic"
            except ImportError:
                return None, None, "检测到 ANTHROPIC_API_KEY，但未安装 anthropic SDK"
        else:
            return None, None, "未设置 OPENAI_API_KEY / ANTHROPIC_API_KEY"
        model = os.getenv("LLM_MODEL", DEFAULT_MODELS[provider])
        if is_offline():
            return None, model, f"离线模式（QTC_OFFLINE=1），只读缓存（{provider}:{model}）"
        return provider, model, f"{provider}:{model}"

    @property
    def available(self) -> bool:
        return self.provider is not None

    def describe(self) -> str:
        if self.available:
            return f"🤖 LLM 可用：{self.reason}（本进程最多 {self.max_calls} 次请求，结果缓存到 {CACHE.name}/）"
        return f"🧩 LLM 不可用：{self.reason} → 使用降级路径（规则 / 词典 / 预置响应）"

    def _key(self, system: str, prompt: str) -> str:
        raw = json.dumps([self.model or "", system, prompt], ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def json(self, prompt: str, system: str = "你是严谨的量化研究助手。只输出 JSON。", max_tokens: int = 2000):
        """返回解析后的 JSON；不可用 / 超预算 / 失败时返回 None"""
        path = CACHE / f"{self._key(system, prompt)}.json"
        if path.exists():
            self.cache_hits += 1
            return json.loads(path.read_text(encoding="utf-8"))["parsed"]
        if not self.available or LLMClient.calls_made >= self.max_calls:
            return None
        LLMClient.calls_made += 1
        try:
            text = self._call(system, prompt, max_tokens)
        except Exception as e:  # 网络、鉴权、限流等错误统一降级
            self.errors.append(f"{type(e).__name__}: {str(e)[:120]}")
            return None
        parsed = parse_json(text)
        if parsed is not None:
            CACHE.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"model": self.model, "prompt": prompt, "raw": text, "parsed": parsed},
                                       ensure_ascii=False, indent=1), encoding="utf-8")
        return parsed

    def _call(self, system: str, prompt: str, max_tokens: int) -> str:
        if self.provider == "openai":
            from openai import OpenAI

            client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL") or None, timeout=60, max_retries=1)
            resp = client.chat.completions.create(
                model=self.model, temperature=0, max_tokens=max_tokens,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content
        import anthropic

        client = anthropic.Anthropic(timeout=120.0, max_retries=1)
        resp = client.messages.create(
            model=self.model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        if resp.stop_reason == "refusal":
            raise RuntimeError("模型拒绝回答（stop_reason=refusal）")
        return "".join(b.text for b in resp.content if b.type == "text")
