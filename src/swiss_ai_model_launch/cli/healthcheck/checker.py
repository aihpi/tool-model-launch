import os

import httpx

from swiss_ai_model_launch.cli.healthcheck.model_health import ModelHealth

_MESSAGE = {"role": "user", "content": 'Say the word "Hello". Nothing more.'}
# The gateway the served model is probed through. Another site sets
# SML_HEALTH_CHECK_URL, and SML_HEALTH_MODEL_PREFIX when its gateway exposes the
# served name under a prefix (e.g. LiteLLM's "hosted_vllm/").
_HEALTH_CHECK_URL = os.environ.get("SML_HEALTH_CHECK_URL") or "https://api.swissai.svc.cscs.ch/v1/chat/completions"
_MODEL_PREFIX = os.environ.get("SML_HEALTH_MODEL_PREFIX", "")
_TIMEOUT_SECONDS = 10
_MAX_TOKENS = 16


async def check_model_health(model_name: str, api_key: str) -> ModelHealth:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                _HEALTH_CHECK_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                json={
                    "model": f"{_MODEL_PREFIX}{model_name}",
                    "messages": [_MESSAGE],
                    "stream": False,
                    "max_tokens": _MAX_TOKENS,
                },
                timeout=_TIMEOUT_SECONDS,
            )
        if response.is_success:
            return ModelHealth.HEALTHY
        return ModelHealth.NOT_RESPONDING
    except (httpx.TransportError, httpx.TimeoutException):
        return ModelHealth.ERROR
