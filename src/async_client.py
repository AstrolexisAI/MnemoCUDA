"""MnemoCUDA Async HTTP Client — talks to mnemo_server over HTTP/SSE.

Connects to the MnemoCUDA inference server (default http://localhost:8095)
which runs expert-streaming MoE inference on NVIDIA GPUs.

Protocol:
    POST /v1/completions  {"prompt":"...", "max_tokens":N, "temperature":T, "stream":true/false}
    Streaming:     SSE events — data: {"token":"...", "done":false/true} then data: [DONE]
    Non-streaming: JSON       — {"text":"...", "tokens":N, "tok_per_sec":X}

Usage:
    from core.mnemo_cuda.async_client import MnemoCudaClient

    client = MnemoCudaClient()
    # Streaming
    async for token in client.generate_stream("Hola, KULVEX"):
        print(token, end="", flush=True)

    # Non-streaming
    text = await client.generate("Hola, KULVEX")

    # Chat (multi-turn with ChatML)
    messages = [
        {"role": "system", "content": "Eres KULVEX, un asistente."},
        {"role": "user", "content": "Hola"},
    ]
    async for token in client.chat_stream(messages):
        print(token, end="", flush=True)
"""

from __future__ import annotations

import json
import logging
import re
from typing import AsyncIterator

import httpx

logger = logging.getLogger("kulvex.mnemo_cuda")

# Default server URL — can be overridden via config or constructor
_DEFAULT_BASE_URL = "http://localhost:8095"

# Strip <think>...</think> reasoning blocks (Qwen3 chain-of-thought)
_THINK_BLOCK_RE = re.compile(r"<think>[\s\S]*?</think>\s*", re.IGNORECASE)
_THINK_UNCLOSED_RE = re.compile(r"<think>[\s\S]*$", re.IGNORECASE)
_THINK_TAG_RE = re.compile(r"</?think>\s*", re.IGNORECASE)


def _strip_think(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from model output."""
    text = _THINK_BLOCK_RE.sub("", text)
    text = _THINK_UNCLOSED_RE.sub("", text)
    text = _THINK_TAG_RE.sub("", text)
    return text.lstrip(": \n")


def build_chatml(messages: list[dict], add_generation_prompt: bool = True) -> str:
    """Format a list of chat messages as a ChatML prompt for Qwen3.

    Each message dict must have "role" (system/user/assistant) and "content".

    When add_generation_prompt is True (default), appends the opening
    assistant turn so the model starts generating immediately.

    NOTE: The MnemoCUDA server currently wraps single prompts in ChatML
    internally. For multi-turn conversations, pass raw_prompt=true in the
    request payload so the server skips its own wrapping and uses this
    pre-built ChatML verbatim.

    Example output:
        <|im_start|>system
        You are KULVEX.<|im_end|>
        <|im_start|>user
        Hello<|im_end|>
        <|im_start|>assistant
        <think>

        </think>

    """
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

    prompt = "\n".join(parts)

    if add_generation_prompt:
        # Open the assistant turn — model generates from here.
        # Include <think>\n\n</think>\n\n to skip reasoning and go straight
        # to content output (reasoning_budget=0 equivalent).
        prompt += "\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    return prompt


class MnemoCudaClient:
    """Async HTTP client for the MnemoCUDA inference server.

    Follows the same structural pattern as OllamaClient for consistency
    across the KULVEX codebase.
    """

    def __init__(self, base_url: str | None = None):
        self._base_url = (base_url or self._config_url()).rstrip("/")
        self._client: httpx.AsyncClient | None = None

    @staticmethod
    def _config_url() -> str:
        """Try to read URL from KULVEX config, fall back to default."""
        try:
            from core.config import settings
            url = getattr(settings, "mnemo_cuda_base_url", None)
            if url:
                return url
        except Exception:
            pass
        return _DEFAULT_BASE_URL

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=300.0,   # MoE expert streaming can be slow on first request
                    write=10.0,
                    pool=10.0,
                ),
            )
        return self._client

    async def close(self) -> None:
        """Close the underlying httpx client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ── Raw completion endpoints ──────────────────────────────

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        raw_prompt: bool = False,
        strip_think: bool = True,
    ) -> str:
        """Generate a completion (non-streaming). Returns the full text.

        Args:
            prompt: The prompt string.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            raw_prompt: If True, tells the server to skip ChatML wrapping
                        (prompt is already formatted).
            strip_think: If True, strip <think> blocks from output.
        """
        payload: dict = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if raw_prompt:
            payload["raw_prompt"] = True

        resp = await self.client.post("/v1/completions", json=payload)
        resp.raise_for_status()

        data = resp.json()
        text = data.get("text", "")
        if strip_think:
            text = _strip_think(text)
        return text

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        raw_prompt: bool = False,
        strip_think: bool = True,
    ) -> AsyncIterator[str]:
        """Generate a completion with SSE streaming. Yields tokens.

        Args:
            prompt: The prompt string.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            raw_prompt: If True, tells the server to skip ChatML wrapping.
            strip_think: If True, filter out <think> block tokens.
        """
        payload: dict = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if raw_prompt:
            payload["raw_prompt"] = True

        in_think_block = False

        async with self.client.stream("POST", "/v1/completions", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue

                # SSE format: "data: {...}" or "data: [DONE]"
                if line.startswith("data: "):
                    line = line[6:]
                if line.strip() == "[DONE]":
                    return

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                token = data.get("token", "")
                if not token:
                    if data.get("done", False):
                        return
                    continue

                # Filter <think> blocks in streaming mode
                if strip_think:
                    if "<think>" in token:
                        in_think_block = True
                        continue
                    if "</think>" in token:
                        in_think_block = False
                        continue
                    if in_think_block:
                        continue

                yield token

                if data.get("done", False):
                    return

    # ── Chat endpoints (ChatML formatting) ────────────────────

    async def chat(
        self,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.7,
        strip_think: bool = True,
    ) -> str:
        """Chat completion from a list of messages (non-streaming).

        Converts messages to ChatML format and sends as a raw prompt.
        """
        prompt = build_chatml(messages, add_generation_prompt=True)
        return await self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            raw_prompt=True,
            strip_think=strip_think,
        )

    async def chat_stream(
        self,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.7,
        strip_think: bool = True,
    ) -> AsyncIterator[str]:
        """Chat completion with SSE streaming. Yields tokens.

        Converts messages to ChatML format and streams the response.
        """
        prompt = build_chatml(messages, add_generation_prompt=True)
        async for token in self.generate_stream(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            raw_prompt=True,
            strip_think=strip_think,
        ):
            yield token

    # ── Health check ──────────────────────────────────────────

    async def health(self) -> dict:
        """Check if the MnemoCUDA server is reachable and responding.

        Sends a minimal 1-token generation to verify the engine is loaded.
        Returns a dict with status info, or raises on failure.
        """
        try:
            resp = await self.client.post(
                "/v1/completions",
                json={
                    "prompt": "ping",
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "stream": False,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "status": "ok",
                "server": self._base_url,
                "tokens_generated": data.get("tokens", 0),
                "tok_per_sec": data.get("tok_per_sec", 0),
            }
        except httpx.ConnectError:
            logger.warning(f"MnemoCUDA server unreachable at {self._base_url}")
            return {"status": "unreachable", "server": self._base_url}
        except Exception as e:
            logger.error(f"MnemoCUDA health check failed: {e}")
            return {"status": "error", "server": self._base_url, "error": str(e)}

    async def is_healthy(self) -> bool:
        """Quick boolean health check."""
        result = await self.health()
        return result.get("status") == "ok"

    # ── Context manager ───────────────────────────────────────

    async def __aenter__(self) -> MnemoCudaClient:
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()
