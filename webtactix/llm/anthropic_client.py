# webtactix/llm/anthropic_client
"""Anthropic Claude client with the same interface as OpenAICompatClient.

Wraps the Anthropic SDK so that WebTactix can use Claude models
without any changes to agents, workflows, or runners.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import anthropic


@dataclass(frozen=True)
class AnthropicConfig:
    """Configuration for the Anthropic Claude client."""

    api_key: str
    model: str = "claude-sonnet-4-6"
    temperature: float = 0.0
    max_tokens: int = 16384
    timeout_s: float = 90.0
    base_url: str | None = None


def _load_api_key_from_xlm_core() -> tuple[str, str | None]:
    """Load Anthropic API key (and optional base URL) from environment or xlm_core.

    Returns:
        A ``(api_key, base_url)`` tuple.  ``base_url`` is ``None`` when calling
        the Anthropic API directly (no Azure Foundry proxy).
    """
    base_url = os.environ.get("ANTHROPIC_BASE_URL") or None
    provider = os.environ.get("LLM_PROVIDER", "").lower()

    if key := os.environ.get("ANTHROPIC_API_KEY"):
        label = "Azure Foundry" if provider == "foundry" else "environment"
        print(f"[LLM] ANTHROPIC_API_KEY loaded from {label}")
        if base_url:
            print(f"[LLM] ANTHROPIC_BASE_URL = {base_url}")
        return key, base_url

    print("[LLM] ANTHROPIC_API_KEY not in env, trying xlm_core Key Vault...")
    try:
        from xlm_core import KeyManagementClient
        kc = KeyManagementClient()
        key = kc.get_secret("ANTHROPIC-API-KEY")
        if not key:
            raise ValueError("Secret 'ANTHROPIC-API-KEY' not found in vault")
        os.environ["ANTHROPIC_API_KEY"] = key
        os.environ.pop("ANTHROPIC_BASE_URL", None)
        print("[LLM] ANTHROPIC_API_KEY fetched from Key Vault (direct API)")
        return key, None
    except Exception as exc:
        raise RuntimeError(
            "Cannot load ANTHROPIC_API_KEY. Set ANTHROPIC_API_KEY (and "
            "ANTHROPIC_BASE_URL for Azure Foundry) in your .env file, or "
            "ensure xlm_core Key Vault access is configured. "
            f"Error: {exc}"
        ) from exc


class AnthropicClient:
    """Drop-in replacement for OpenAICompatClient using Claude.

    Exposes the same ``chat_text`` and ``chat_json`` async methods
    so the rest of WebTactix can use it without modification.
    """

    def __init__(self, cfg: AnthropicConfig) -> None:
        self.cfg = cfg
        kwargs: dict[str, Any] = {
            "api_key": cfg.api_key,
            "timeout": cfg.timeout_s,
        }
        if cfg.base_url:
            kwargs["base_url"] = cfg.base_url
        self._client = anthropic.AsyncAnthropic(**kwargs)

    async def chat_text(
        self,
        *,
        system: str,
        user: str,
        temperature: Optional[float] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Send a system+user prompt to Claude, return (text, usage)."""
        temp = (
            self.cfg.temperature
            if temperature is None
            else float(temperature)
        )

        resp = None
        last_err = None
        for _ in range(3):
            try:
                resp = await self._client.messages.create(
                    model=self.cfg.model,
                    max_tokens=self.cfg.max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                    extra_body={"temperature": temp},
                )
                break
            except Exception as exc:
                print("[LLM ERR]", exc)
                last_err = exc

        if resp is None:
            raise RuntimeError(
                f"Claude call failed after 3 retries: {last_err}"
            )

        output_text = ""
        for block in resp.content:
            if block.type == "text":
                output_text += block.text

        usage: Dict[str, Any] = {
            "prompt_tokens": resp.usage.input_tokens,
            "completion_tokens": resp.usage.output_tokens,
            "total_tokens": (
                resp.usage.input_tokens + resp.usage.output_tokens
            ),
            "estimated": False,
            "model": self.cfg.model,
        }

        return output_text.strip(), usage

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract a JSON object/array from LLM text.

        Handles all common response formats:
        - Pure JSON
        - JSON wrapped in markdown code fences
        - Preamble text before a fenced JSON block
        - Preamble text before raw JSON (``{`` or ``[``)
        """
        s = text.strip()
        if not s:
            return ""

        # 1) Try to find JSON inside ```...``` fences.
        fence_start = s.find("```")
        if fence_start != -1:
            after_fence = s[fence_start + 3:]
            newline = after_fence.find("\n")
            if newline != -1:
                body_start = after_fence[newline + 1:]
            else:
                body_start = after_fence
            fence_end = body_start.find("```")
            if fence_end != -1:
                return body_start[:fence_end].strip()
            return body_start.strip()

        # 2) No fences — find the first { or [ that starts JSON.
        for i, ch in enumerate(s):
            if ch in ("{", "["):
                candidate = s[i:]
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    pass
                break

        # 3) Fallback: return the whole string and let the
        #    caller handle the parse error.
        return s

    async def chat_json(
        self,
        *,
        system: str,
        user: str,
        temperature: Optional[float] = None,
        max_retries: int = 2,
    ) -> Tuple[Union[Dict[str, Any], list], Dict[str, Any]]:
        """Send a prompt and parse the response as JSON.

        Retries up to ``max_retries`` times when the response is
        empty or contains invalid JSON.
        """
        last_err: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            text, usage = await self.chat_text(
                system=system, user=user, temperature=temperature,
            )

            s = self._extract_json(text)

            if not s:
                last_err = ValueError("LLM returned empty response")
                print(
                    f"[LLM] empty response "
                    f"(attempt {attempt}/{max_retries})"
                )
                continue

            try:
                obj = json.loads(s)
            except json.JSONDecodeError as exc:
                last_err = exc
                print(
                    f"[LLM] JSON parse error "
                    f"(attempt {attempt}/{max_retries}): {exc}"
                )
                print(f"[LLM] raw text: {text[:500]}")
                continue

            if not isinstance(obj, (dict, list)):
                last_err = ValueError(
                    f"JSON root must be dict or list, "
                    f"got {type(obj)}"
                )
                print(
                    f"[LLM] unexpected root type "
                    f"(attempt {attempt}/{max_retries})"
                )
                continue

            return obj, usage

        raise RuntimeError(
            f"chat_json failed after {max_retries} attempts: "
            f"{last_err}"
        ) from last_err
