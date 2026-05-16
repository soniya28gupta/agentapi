"""Tests for empty/missing/malformed choices handling in OpenAICompatibleProvider.

Issue #33: OpenAI-compatible provider should raise AgentProviderError with a
clear message instead of leaking raw KeyError / IndexError / TypeError.

Only the non-streaming chat() method is tested here; stream() already handles
empty choices gracefully via `chunk.get("choices") or []`.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentapi.errors import AgentProviderError
from agentapi.providers.openai_compatible import OpenAICompatibleProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_provider() -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        api_key="test-key",
        model="test-model",
        base_url="https://api.example.com/v1",
    )


def _mock_http_response(body: dict) -> MagicMock:
    """Return a mock httpx.Response that yields *body* from .json()."""
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = body
    response.raise_for_status = MagicMock()  # no-op — 200 OK
    return response


def _patch_client(response_body: dict):
    """Context manager that patches httpx.AsyncClient.post to return a mock response."""
    mock_response = _mock_http_response(response_body)
    mock_post = AsyncMock(return_value=mock_response)
    mock_client = MagicMock()
    mock_client.post = mock_post
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return patch("httpx.AsyncClient", return_value=mock_client)


# ---------------------------------------------------------------------------
# 1. Happy path — valid response
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chat_valid_response():
    """A well-formed provider response is parsed and returned without error."""
    body = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Hello, world!",
                    "tool_calls": None,
                }
            }
        ]
    }
    provider = _make_provider()
    with _patch_client(body):
        result = await provider.chat([{"role": "user", "content": "Hi"}])

    assert result.content == "Hello, world!"
    assert result.tool_calls == []
    assert result.raw_message["role"] == "assistant"


# ---------------------------------------------------------------------------
# 2. Missing "choices" key entirely
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chat_missing_choices_key():
    """Response JSON with no 'choices' key raises AgentProviderError (502)."""
    body = {"id": "chatcmpl-abc", "object": "chat.completion"}  # no "choices"
    provider = _make_provider()
    with _patch_client(body):
        with pytest.raises(AgentProviderError) as exc_info:
            await provider.chat([{"role": "user", "content": "Hi"}])

    err = exc_info.value
    assert err.status_code == 502
    assert "empty or missing" in str(err).lower() or "choices" in str(err)


# ---------------------------------------------------------------------------
# 3. choices is null / None
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chat_null_choices():
    """Response JSON with choices=null raises AgentProviderError (502)."""
    body = {"choices": None}
    provider = _make_provider()
    with _patch_client(body):
        with pytest.raises(AgentProviderError) as exc_info:
            await provider.chat([{"role": "user", "content": "Hi"}])

    err = exc_info.value
    assert err.status_code == 502
    assert "choices" in str(err)


# ---------------------------------------------------------------------------
# 4. choices is an empty list
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chat_empty_choices_list():
    """Response JSON with choices=[] raises AgentProviderError (502)."""
    body = {"choices": []}
    provider = _make_provider()
    with _patch_client(body):
        with pytest.raises(AgentProviderError) as exc_info:
            await provider.chat([{"role": "user", "content": "Hi"}])

    err = exc_info.value
    assert err.status_code == 502
    assert "choices" in str(err)


# ---------------------------------------------------------------------------
# 5. choices[0] has no "message" field
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chat_missing_message_in_first_choice():
    """choices[0] entry without a 'message' key raises AgentProviderError (502)."""
    body = {"choices": [{"index": 0, "finish_reason": "stop"}]}  # no "message"
    provider = _make_provider()
    with _patch_client(body):
        with pytest.raises(AgentProviderError) as exc_info:
            await provider.chat([{"role": "user", "content": "Hi"}])

    err = exc_info.value
    assert err.status_code == 502
    assert "message" in str(err)


# ---------------------------------------------------------------------------
# Runner (optional — pytest discovers tests automatically)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
