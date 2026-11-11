"""The ONE model plug. Every LLM call in the Lab goes through `get_llm(cfg).complete(...)`.

Backends:
  fake               scripted replies, for tests and dry runs
  ollama (default)   local, free: POST {base_url}/api/chat
  openai_compatible  any server speaking the OpenAI chat API (key from an env var)
  anthropic          official `anthropic` SDK (key from an env var; default model claude-opus-5-5)

Never hard-code keys: the config names an environment variable, nothing more.
"""

import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx


@dataclass(frozen=True)
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass(frozen=True)
class LLMConfig:
    backend: str = "ollama"
    model: str = "qwen2.5:3b"
    base_url: str = "http://localhost:11434"
    num_ctx: int = 8192  # Ollama only. Its small default silently cuts long prompts.
    api_key_env: str | None = None
    max_tokens: int = 4096
    timeout_s: float = 600.0
    temperature: float = 0.2
    scripted: list[str] = field(default_factory=list)  # fake backend only


class LLM(Protocol):
    name: str

    def complete(self, messages: Sequence[Message], *, json_mode: bool = False) -> str: ...


class LLMError(RuntimeError):
    pass


# --- helpers -----------------------------------------------------------------------------


def split_system(messages: Sequence[Message]) -> tuple[str, list[dict[str, str]]]:
    system = "\n\n".join(m.content for m in messages if m.role == "system")
    rest = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]
    return system, rest


_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model reply (fenced, or bare, or with prose)."""
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    m = _JSON_FENCE.search(text)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass  # a broken fenced block: fall through and scan for any other object
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    raise LLMError("no JSON object found in model reply")


def _api_key(cfg: LLMConfig) -> str | None:
    if not cfg.api_key_env:
        return None
    key = os.environ.get(cfg.api_key_env)
    if not key:
        raise LLMError(f"environment variable {cfg.api_key_env} is not set")
    return key


# --- backends ----------------------------------------------------------------------------


class FakeLLM:
    """Returns scripted replies in order; records every prompt it saw."""

    name = "fake"

    def __init__(self, scripted: Sequence[str]) -> None:
        self._replies = list(scripted)
        self.calls: list[list[Message]] = []

    def complete(self, messages: Sequence[Message], *, json_mode: bool = False) -> str:
        self.calls.append(list(messages))
        if not self._replies:
            raise LLMError("FakeLLM ran out of scripted replies")
        return self._replies.pop(0)


class OllamaLLM:
    name = "ollama"

    def __init__(self, cfg: LLMConfig, client: httpx.Client | None = None) -> None:
        self.cfg = cfg
        self._client = client or httpx.Client(timeout=cfg.timeout_s)

    def complete(self, messages: Sequence[Message], *, json_mode: bool = False) -> str:
        body: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {
                "temperature": self.cfg.temperature,
                "num_predict": self.cfg.max_tokens,
                "num_ctx": self.cfg.num_ctx,
            },
        }
        if json_mode:
            body["format"] = "json"
        try:
            r = self._client.post(f"{self.cfg.base_url.rstrip('/')}/api/chat", json=body)
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"ollama request failed: {exc}") from exc
        data = r.json()
        content = data.get("message", {}).get("content")
        if not isinstance(content, str):
            raise LLMError(f"unexpected ollama reply: {str(data)[:200]}")
        return content


class OpenAICompatibleLLM:
    name = "openai_compatible"

    def __init__(self, cfg: LLMConfig, client: httpx.Client | None = None) -> None:
        self.cfg = cfg
        self._client = client or httpx.Client(timeout=cfg.timeout_s)

    def complete(self, messages: Sequence[Message], *, json_mode: bool = False) -> str:
        headers = {"Content-Type": "application/json"}
        key = _api_key(self.cfg)
        if key:
            headers["Authorization"] = f"Bearer {key}"
        body: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        try:
            r = self._client.post(
                f"{self.cfg.base_url.rstrip('/')}/chat/completions", json=body, headers=headers
            )
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"openai-compatible request failed: {exc}") from exc
        data = r.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected reply: {str(data)[:200]}") from exc
        return str(content)


class AnthropicLLM:
    """Official SDK. Thinking is adaptive by default on current models; we leave it on."""

    name = "anthropic"

    def __init__(self, cfg: LLMConfig) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise LLMError("install the optional dependency: uv sync --extra anthropic") from exc
        self.cfg = cfg
        self._client = anthropic.Anthropic(api_key=_api_key(cfg), timeout=cfg.timeout_s)

    def complete(self, messages: Sequence[Message], *, json_mode: bool = False) -> str:
        system, rest = split_system(messages)
        if json_mode:
            system = (system + "\n\nReply with a single JSON object and nothing else.").strip()
        response = self._client.messages.create(
            model=self.cfg.model or "claude-opus-5-5",
            max_tokens=self.cfg.max_tokens,
            system=system or anthropic_not_given(),
            messages=rest,
        )
        if response.stop_reason == "refusal":
            raise LLMError("the model declined this request (stop_reason=refusal)")
        return "".join(getattr(b, "text", "") for b in response.content if b.type == "text")


def anthropic_not_given() -> Any:
    import anthropic

    return anthropic.NOT_GIVEN


def get_llm(cfg: LLMConfig) -> LLM:
    """The ONLY place a backend is chosen."""
    if cfg.backend == "fake":
        return FakeLLM(cfg.scripted)
    if cfg.backend == "ollama":
        return OllamaLLM(cfg)
    if cfg.backend == "openai_compatible":
        return OpenAICompatibleLLM(cfg)
    if cfg.backend == "anthropic":
        return AnthropicLLM(cfg)
    raise LLMError(f"unknown llm backend {cfg.backend!r}")
