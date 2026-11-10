"""LLM provider interface with OpenAI, Ollama (local, free) and a deterministic fake for tests."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Protocol

import httpx

from app.config import get_settings


class ChatModel(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str: ...


class OpenAIChat:
    def __init__(self, api_key: str, model: str):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model

    def complete(self, messages: list[dict[str, str]]) -> str:
        response = self._client.chat.completions.create(
            model=self._model, messages=messages, temperature=0
        )
        return response.choices[0].message.content or ""


class OllamaChat:
    def __init__(self, base_url: str, model: str):
        self._base_url = base_url.rstrip("/")
        self._model = model

    def complete(self, messages: list[dict[str, str]]) -> str:
        response = httpx.post(
            f"{self._base_url}/api/chat",
            json={
                "model": self._model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=180,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]


class FakeChat:
    """Deterministic offline model. Follows the same two prompt formats the agent uses."""

    def complete(self, messages: list[dict[str, str]]) -> str:
        system = messages[0]["content"]
        user = messages[-1]["content"]
        if "Respond ONLY with JSON" in system:
            tool_results = int(re.search(r"Tool results so far: (\d+)", user).group(1))
            question = re.search(r"Question: (.*)", user).group(1).strip()
            if tool_results == 0:
                return f'{{"action": "search", "query": {_json_str(question)}}}'
            return '{"action": "answer"}'
        # Answer step: summarise the first sentence of each numbered source.
        sources = re.findall(r"^\[(\d+)\] (.*)$", user, flags=re.MULTILINE)
        if not sources:
            return "I could not find this in the documents you have access to."
        lines = []
        for number, text in sources[:3]:
            sentence = re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0][:300]
            lines.append(f"{sentence} [{number}]")
        return "Based on the documents: " + " ".join(lines)


def _json_str(value: str) -> str:
    import json

    return json.dumps(value)


@lru_cache
def get_llm() -> ChatModel:
    settings = get_settings()
    provider = settings.llm_provider
    if provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        return OpenAIChat(settings.openai_api_key, settings.openai_model)
    if provider == "ollama":
        return OllamaChat(settings.ollama_base_url, settings.ollama_model)
    if provider == "fake":
        return FakeChat()
    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")
