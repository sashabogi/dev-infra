"""Model routing with failover, circuit breaking, and format transformation."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from daneel.quality_gate import QualityResult, score_response
from daneel.safety import check_safety, scrub_credentials

logger = logging.getLogger("daneel.router")


@dataclass
class ProviderHealth:
    name: str
    consecutive_failures: int = 0
    total_requests: int = 0
    total_successes: int = 0
    total_failures: int = 0
    avg_latency_ms: float = 0.0
    _latency_sum: float = 0.0
    circuit_open_until: float = 0.0

    @property
    def is_available(self) -> bool:
        if self.circuit_open_until > 0 and time.time() < self.circuit_open_until:
            return False
        return True

    def record_success(self, latency_ms: float) -> None:
        self.consecutive_failures = 0
        self.total_requests += 1
        self.total_successes += 1
        self._latency_sum += latency_ms
        self.avg_latency_ms = self._latency_sum / self.total_successes

    def record_failure(self) -> None:
        self.consecutive_failures += 1
        self.total_requests += 1
        self.total_failures += 1
        if self.consecutive_failures >= 3:
            self.circuit_open_until = time.time() + 60.0
            logger.warning(
                "Circuit breaker OPEN for %s (3 consecutive failures, "
                "cooling off 60s)",
                self.name,
            )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "available": self.is_available,
            "consecutive_failures": self.consecutive_failures,
            "total_requests": self.total_requests,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "circuit_open": not self.is_available,
        }


@dataclass
class RoutingResult:
    provider: str
    model: str
    response: dict[str, Any]
    tokens_in: int
    tokens_out: int
    cost_usd: float
    quality: QualityResult
    latency_ms: float
    status: str  # success | failed | rejected
    error: str | None = None
    role: str | None = None


class Router:
    def __init__(self, config: dict):
        self._config = config
        self._providers = config.get("providers", {})
        self._failover_chain: list[str] = config.get("failover_chain", [])
        self._quality_threshold: float = config.get("quality_gate", {}).get(
            "threshold", 0.5
        )
        self._health: dict[str, ProviderHealth] = {
            name: ProviderHealth(name=name) for name in self._providers
        }
        self._api_keys: dict[str, str] = {}
        self._client: httpx.AsyncClient | None = None

    def set_api_keys(self, keys: dict[str, str]) -> None:
        self._api_keys = keys

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=120.0)
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def get_provider_status(self) -> dict[str, dict]:
        return {name: h.to_dict() for name, h in self._health.items()}

    async def route_request(
        self,
        messages: list[dict],
        model: str | None = None,
        target_format: str = "openai",
        role: str | None = None,
        target_provider: str | None = None,
        **kwargs: Any,
    ) -> RoutingResult:
        """Route a request through the failover chain.

        When target_provider is set (Seldon passthrough mode), bypass the
        failover chain and call the specified provider directly.  Quality
        gate, credential scrubbing, and cost tracking still apply.
        """
        # Safety check
        safety = check_safety(messages)
        if not safety.safe:
            return RoutingResult(
                provider="safety_block",
                model="none",
                response=_error_response(
                    f"Request blocked: dangerous patterns detected: "
                    f"{', '.join(safety.blocked_patterns)}",
                    target_format,
                ),
                tokens_in=0,
                tokens_out=0,
                cost_usd=0.0,
                quality=QualityResult(score=0.0, passed=False),
                latency_ms=0.0,
                status="blocked",
                error="Safety block",
                role=role,
            )

        # --- Seldon passthrough mode ---
        if target_provider:
            if target_provider not in self._providers:
                # Unknown provider — caller (server.py) handles forwarding
                return RoutingResult(
                    provider=target_provider,
                    model=model or "unknown",
                    response=_error_response(
                        f"Provider '{target_provider}' not in Daneel config",
                        target_format,
                    ),
                    tokens_in=0,
                    tokens_out=0,
                    cost_usd=0.0,
                    quality=QualityResult(score=0.0, passed=False),
                    latency_ms=0.0,
                    status="unknown_provider",
                    error=f"Provider '{target_provider}' not configured",
                    role=role,
                )

            health = self._health.get(target_provider)
            if health and not health.is_available:
                logger.warning(
                    "Role-directed request (role=%s) to %s blocked: "
                    "circuit breaker open",
                    role,
                    target_provider,
                )
                return RoutingResult(
                    provider=target_provider,
                    model=self._providers[target_provider].get("model", "unknown"),
                    response=_error_response(
                        f"Provider '{target_provider}' circuit breaker is open. "
                        f"Role-directed requests do not failover.",
                        target_format,
                    ),
                    tokens_in=0,
                    tokens_out=0,
                    cost_usd=0.0,
                    quality=QualityResult(score=0.0, passed=False),
                    latency_ms=0.0,
                    status="failed",
                    error=f"{target_provider}: circuit breaker open (role-directed)",
                    role=role,
                )

            logger.info(
                "Role-directed request: role=%s provider=%s model=%s",
                role,
                target_provider,
                model or self._providers[target_provider].get("model"),
            )
            result = await self._try_provider(
                target_provider, messages, target_format,
                model_override=model, **kwargs,
            )
            result.role = role
            return result

        # --- Default mode: cost-route through failover chain ---

        # If explicit model is requested, try to find matching provider
        if model:
            for name, pconfig in self._providers.items():
                if pconfig.get("model") == model:
                    result = await self._try_provider(
                        name, messages, target_format, **kwargs
                    )
                    if result.status == "success":
                        result.role = role
                        return result
                    break

        # Walk the failover chain
        errors: list[str] = []
        for provider_name in self._failover_chain:
            if provider_name not in self._providers:
                continue

            health = self._health.get(provider_name)
            if health and not health.is_available:
                errors.append(f"{provider_name}: circuit breaker open")
                continue

            result = await self._try_provider(
                provider_name, messages, target_format, **kwargs
            )
            if result.status == "success":
                result.role = role
                return result

            errors.append(f"{provider_name}: {result.error}")

        return RoutingResult(
            provider="none",
            model="none",
            response=_error_response(
                f"All providers failed. Errors: {'; '.join(errors)}",
                target_format,
            ),
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.0,
            quality=QualityResult(score=0.0, passed=False),
            latency_ms=0.0,
            status="failed",
            error="; ".join(errors),
            role=role,
        )

    async def _try_provider(
        self,
        provider_name: str,
        messages: list[dict],
        target_format: str,
        model_override: str | None = None,
        **kwargs: Any,
    ) -> RoutingResult:
        """Attempt a single provider call."""
        pconfig = self._providers[provider_name]
        health = self._health[provider_name]
        model = model_override or pconfig["model"]

        # Build the OpenAI-format request body
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        # Pass through common params
        for k in ("temperature", "max_tokens", "top_p", "stop"):
            if k in kwargs and kwargs[k] is not None:
                body[k] = kwargs[k]

        # Resolve API key
        key_env = pconfig.get("api_key_env", "")
        api_key = self._api_keys.get(key_env, "") if key_env else ""

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        endpoint = pconfig["endpoint"]
        start = time.monotonic()

        try:
            client = await self._get_client()
            resp = await client.post(endpoint, json=body, headers=headers)
            latency_ms = (time.monotonic() - start) * 1000

            if resp.status_code != 200:
                error_text = resp.text[:500]
                health.record_failure()
                logger.warning(
                    "Provider %s returned %d: %s",
                    provider_name,
                    resp.status_code,
                    error_text,
                )
                return RoutingResult(
                    provider=provider_name,
                    model=model,
                    response={},
                    tokens_in=0,
                    tokens_out=0,
                    cost_usd=0.0,
                    quality=QualityResult(score=0.0, passed=False),
                    latency_ms=latency_ms,
                    status="failed",
                    error=f"HTTP {resp.status_code}: {error_text}",
                )

            data = resp.json()

            # Extract response text for quality scoring
            response_text = _extract_text(data)

            # Quality gate
            quality = score_response(response_text, self._quality_threshold)
            if not quality.passed:
                health.record_failure()
                logger.warning(
                    "Provider %s failed quality gate (%.2f < %.2f): %s",
                    provider_name,
                    quality.score,
                    self._quality_threshold,
                    "; ".join(quality.details),
                )
                return RoutingResult(
                    provider=provider_name,
                    model=model,
                    response=data,
                    tokens_in=_get_usage(data, "input"),
                    tokens_out=_get_usage(data, "output"),
                    cost_usd=_calc_cost(data, pconfig),
                    quality=quality,
                    latency_ms=latency_ms,
                    status="rejected",
                    error=f"Quality gate: {'; '.join(quality.details)}",
                )

            # Scrub credentials from response
            scrubbed_text, was_scrubbed = scrub_credentials(response_text)
            if was_scrubbed:
                data = _replace_text(data, scrubbed_text)

            # Transform response format if needed
            if target_format == "anthropic":
                data = _openai_to_anthropic(data)

            tokens_in = _get_usage(data, "input")
            tokens_out = _get_usage(data, "output")
            cost = _calc_cost_raw(
                tokens_in,
                tokens_out,
                pconfig.get("cost_per_million", {}),
            )

            health.record_success(latency_ms)

            return RoutingResult(
                provider=provider_name,
                model=model,
                response=data,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=cost,
                quality=quality,
                latency_ms=latency_ms,
                status="success",
            )

        except httpx.TimeoutException:
            latency_ms = (time.monotonic() - start) * 1000
            health.record_failure()
            return RoutingResult(
                provider=provider_name,
                model=model,
                response={},
                tokens_in=0,
                tokens_out=0,
                cost_usd=0.0,
                quality=QualityResult(score=0.0, passed=False),
                latency_ms=latency_ms,
                status="failed",
                error="Request timed out",
            )
        except Exception as exc:
            latency_ms = (time.monotonic() - start) * 1000
            health.record_failure()
            logger.exception("Provider %s error", provider_name)
            return RoutingResult(
                provider=provider_name,
                model=model,
                response={},
                tokens_in=0,
                tokens_out=0,
                cost_usd=0.0,
                quality=QualityResult(score=0.0, passed=False),
                latency_ms=latency_ms,
                status="failed",
                error=str(exc),
            )


# --- Format transformation helpers ---


def anthropic_to_openai_messages(messages: list[dict]) -> list[dict]:
    """Transform Anthropic Messages API format to OpenAI format."""
    result = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, list):
            # Anthropic uses content blocks: [{"type": "text", "text": "..."}]
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            content = "\n".join(text_parts)

        result.append({"role": role, "content": content})
    return result


def _openai_to_anthropic(data: dict) -> dict:
    """Transform OpenAI response to Anthropic Messages API format."""
    choices = data.get("choices", [])
    text = ""
    if choices:
        msg = choices[0].get("message", {})
        text = msg.get("content", "")

    usage = data.get("usage", {})

    return {
        "id": data.get("id", ""),
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": data.get("model", ""),
        "stop_reason": _map_finish_reason(
            choices[0].get("finish_reason") if choices else None
        ),
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


def _map_finish_reason(reason: str | None) -> str:
    mapping = {
        "stop": "end_turn",
        "length": "max_tokens",
        "content_filter": "end_turn",
    }
    return mapping.get(reason or "", "end_turn")


def _extract_text(data: dict) -> str:
    """Extract text content from OpenAI-format response."""
    choices = data.get("choices", [])
    if not choices:
        return ""
    msg = choices[0].get("message", {})
    return msg.get("content", "") or ""


def _replace_text(data: dict, new_text: str) -> dict:
    """Replace text content in OpenAI-format response."""
    choices = data.get("choices", [])
    if choices and "message" in choices[0]:
        data["choices"][0]["message"]["content"] = new_text
    return data


def _get_usage(data: dict, kind: str) -> int:
    """Extract token usage from response."""
    usage = data.get("usage", {})
    if kind == "input":
        return usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
    return usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)


def _calc_cost(data: dict, pconfig: dict) -> float:
    tokens_in = _get_usage(data, "input")
    tokens_out = _get_usage(data, "output")
    return _calc_cost_raw(tokens_in, tokens_out, pconfig.get("cost_per_million", {}))


def _calc_cost_raw(tokens_in: int, tokens_out: int, cost_config: dict) -> float:
    in_cost = (tokens_in / 1_000_000) * cost_config.get("input", 0)
    out_cost = (tokens_out / 1_000_000) * cost_config.get("output", 0)
    return round(in_cost + out_cost, 8)


def _error_response(message: str, fmt: str) -> dict:
    if fmt == "anthropic":
        return {
            "type": "error",
            "error": {"type": "server_error", "message": message},
        }
    return {
        "error": {"message": message, "type": "server_error", "code": 500}
    }
