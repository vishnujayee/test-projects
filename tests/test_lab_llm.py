"""The model plug: fake backend, JSON extraction, and request shapes for Ollama / OpenAI-compatible."""

import json

import httpx
import pytest

from trivec.lab.llm import (
    FakeLLM,
    LLMConfig,
    LLMError,
    Message,
    OllamaLLM,
    OpenAICompatibleLLM,
    extract_json,
    get_llm,
)


def test_fake_llm_scripted_and_recorded() -> None:
    llm = FakeLLM(["a", "b"])
    assert llm.complete([Message("user", "x")]) == "a"
    assert llm.complete([Message("user", "y")]) == "b"
    assert len(llm.calls) == 2
    with pytest.raises(LLMError):
        llm.complete([Message("user", "z")])


@pytest.mark.parametrize(
    "text",
    [
        '{"a": 1}',
        'Sure! ```json\n{"a": 1}\n```',
        'Here you go: {"a": 1} hope it helps',
        'prefix {"b": {"c": [1,2]}} {"a": 1}',
    ],
)
def test_extract_json_variants(text: str) -> None:
    obj = extract_json(text)
    assert isinstance(obj, dict) and ("a" in obj or "b" in obj)


def test_extract_json_fails_cleanly() -> None:
    with pytest.raises(LLMError):
        extract_json("no json here")


def test_ollama_request_shape() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"message": {"role": "assistant", "content": '{"ok": true}'}}
        )

    cfg = LLMConfig(backend="ollama", model="m", base_url="http://x:11434")
    llm = OllamaLLM(cfg, client=httpx.Client(transport=httpx.MockTransport(handler)))
    out = llm.complete([Message("system", "s"), Message("user", "u")], json_mode=True)
    assert out == '{"ok": true}'
    assert seen["url"] == "http://x:11434/api/chat"
    body = seen["body"]
    assert isinstance(body, dict)
    assert body["model"] == "m" and body["format"] == "json" and body["stream"] is False
    assert body["messages"][0] == {"role": "system", "content": "s"}


def test_openai_compatible_uses_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_LLM_KEY", "secret")
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

    cfg = LLMConfig(
        backend="openai_compatible", model="m", base_url="http://x/v1", api_key_env="TEST_LLM_KEY"
    )
    llm = OpenAICompatibleLLM(cfg, client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert llm.complete([Message("user", "u")]) == "hi"
    assert seen["auth"] == "Bearer secret"


def test_missing_env_key_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NOPE_KEY", raising=False)
    cfg = LLMConfig(backend="openai_compatible", api_key_env="NOPE_KEY")
    llm = OpenAICompatibleLLM(
        cfg, client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    )
    with pytest.raises(LLMError):
        llm.complete([Message("user", "u")])


def test_get_llm_dispatch() -> None:
    assert get_llm(LLMConfig(backend="fake", scripted=["x"])).name == "fake"
    assert get_llm(LLMConfig(backend="ollama")).name == "ollama"
    with pytest.raises(LLMError):
        get_llm(LLMConfig(backend="nope"))
