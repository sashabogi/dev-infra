"""Daneel FastAPI server — transparent LLM inference proxy."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import time

import httpx
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from daneel.costs import CostRecord, CostTracker
from daneel.quality_gate import QualityResult, score_response
from daneel.router import Router, RoutingResult, anthropic_to_openai_messages
from daneel.safety import scrub_credentials

logger = logging.getLogger("daneel.server")

# Module-level state
_config: dict = {}
_router: Router | None = None
_cost_tracker: CostTracker | None = None

# Quality gate statistics
_quality_stats: dict[str, int] = {"passed": 0, "failed": 0, "rejected": 0}


def _load_config() -> dict:
    """Load config from ~/.dev-infra/config.yaml, fallback to project config.example.yaml."""
    paths = [
        Path(os.path.expanduser("~/.dev-infra/config.yaml")),
        Path(__file__).parent.parent / "config.example.yaml",
    ]
    for p in paths:
        if p.exists():
            logger.info("Loading config from %s", p)
            with open(p) as f:
                return yaml.safe_load(f)
    raise FileNotFoundError(
        "No config found. Copy config.example.yaml to ~/.dev-infra/config.yaml"
    )


def _resolve_api_keys(config: dict) -> dict[str, str]:
    """Resolve API keys from environment variables."""
    keys: dict[str, str] = {}
    for _name, pconfig in config.get("providers", {}).items():
        env_var = pconfig.get("api_key_env", "")
        if env_var:
            keys[env_var] = os.environ.get(env_var, "")
    return keys


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown."""
    global _config, _router, _cost_tracker

    load_dotenv()

    _config = _load_config()
    _router = Router(_config)
    _router.set_api_keys(_resolve_api_keys(_config))
    _cost_tracker = CostTracker(_config)

    host = _config.get("daneel", {}).get("host", "0.0.0.0")
    port = _config.get("daneel", {}).get("port", 8889)
    logger.info("Daneel proxy starting on %s:%d", host, port)

    yield

    if _router:
        await _router.close()
    if _cost_tracker:
        _cost_tracker.close()


app = FastAPI(
    title="Daneel Inference Proxy",
    description="Routes LLM inference to the cheapest viable model",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "daneel"}


@app.get("/costs")
async def costs():
    if not _cost_tracker:
        return JSONResponse(status_code=503, content={"error": "not initialized"})
    hours = _config.get("costs", {}).get("rolling_window_hours", 24)
    rolling = _cost_tracker.get_rolling_costs(hours)
    total = _cost_tracker.get_total_saved()
    return {"rolling": rolling, "lifetime": total}


@app.get("/providers")
async def providers():
    if not _router:
        return JSONResponse(status_code=503, content={"error": "not initialized"})
    return {"providers": _router.get_provider_status()}


@app.get("/quality")
async def quality():
    return {"stats": _quality_stats}


@app.post("/rescue")
async def rescue(request: Request):
    """Accept context from hooks and run memory rescue extraction."""
    body = await request.json()
    context = body.get("context", "")
    project = body.get("project")
    session_id = body.get("session_id")

    if not context or len(context) < 100:
        return {"committed": 0, "count": 0, "error": "Context too short"}

    try:
        from rescue.engine import RescueEngine

        engine = RescueEngine(_config)
        committed = await engine.rescue_context(
            context, project=project, session_id=session_id
        )
        engine.close()
        return {"committed": len(committed), "count": len(committed)}
    except Exception as exc:
        logger.exception("Rescue endpoint failed")
        return JSONResponse(
            status_code=500,
            content={"error": str(exc), "committed": 0, "count": 0},
        )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible chat completions endpoint."""
    if not _router or not _cost_tracker:
        return JSONResponse(status_code=503, content={"error": "not initialized"})

    body = await request.json()
    messages = body.get("messages", [])
    model = body.get("model")

    # Seldon role-aware headers
    seldon_role = request.headers.get("X-Seldon-Role")
    target_provider = request.headers.get("X-Provider")

    kwargs: dict[str, Any] = {}
    for k in ("temperature", "max_tokens", "top_p", "stop"):
        if k in body:
            kwargs[k] = body[k]

    result = await _router.route_request(
        messages=messages,
        model=model,
        target_format="openai",
        role=seldon_role,
        target_provider=target_provider,
        **kwargs,
    )

    # Handle unknown provider — forward directly
    if result.status == "unknown_provider" and target_provider:
        result = await _forward_unknown_provider(
            body, target_provider, seldon_role, "openai",
        )

    # Track quality stats
    if result.quality.passed:
        _quality_stats["passed"] += 1
    elif result.status == "rejected":
        _quality_stats["rejected"] += 1
    else:
        _quality_stats["failed"] += 1

    # Record cost
    _cost_tracker.record(
        CostRecord(
            provider=result.provider,
            model=result.model,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            cost_usd=result.cost_usd,
            quality_score=result.quality.score,
            latency_ms=int(result.latency_ms),
            status=result.status,
            role=result.role,
        )
    )

    if result.status in ("failed", "blocked"):
        return JSONResponse(status_code=502, content=result.response)

    return result.response


@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    """Anthropic Messages API compatible endpoint."""
    if not _router or not _cost_tracker:
        return JSONResponse(status_code=503, content={"error": "not initialized"})

    body = await request.json()
    messages = body.get("messages", [])
    model = body.get("model")
    system = body.get("system")

    # Seldon role-aware headers
    seldon_role = request.headers.get("X-Seldon-Role")
    target_provider = request.headers.get("X-Provider")

    # Transform Anthropic format to OpenAI format
    openai_messages = anthropic_to_openai_messages(messages)

    # Prepend system message if provided
    if system:
        if isinstance(system, str):
            openai_messages.insert(0, {"role": "system", "content": system})
        elif isinstance(system, list):
            # Anthropic system can be array of content blocks
            text_parts = []
            for block in system:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            if text_parts:
                openai_messages.insert(
                    0, {"role": "system", "content": "\n".join(text_parts)}
                )

    kwargs: dict[str, Any] = {}
    if "max_tokens" in body:
        kwargs["max_tokens"] = body["max_tokens"]
    if "temperature" in body:
        kwargs["temperature"] = body["temperature"]
    if "top_p" in body:
        kwargs["top_p"] = body["top_p"]
    if "stop_sequences" in body:
        kwargs["stop"] = body["stop_sequences"]

    result = await _router.route_request(
        messages=openai_messages,
        model=model,
        target_format="anthropic",
        role=seldon_role,
        target_provider=target_provider,
        **kwargs,
    )

    # Handle unknown provider — forward directly
    if result.status == "unknown_provider" and target_provider:
        result = await _forward_unknown_provider(
            body, target_provider, seldon_role, "anthropic",
        )

    # Track quality stats
    if result.quality.passed:
        _quality_stats["passed"] += 1
    elif result.status == "rejected":
        _quality_stats["rejected"] += 1
    else:
        _quality_stats["failed"] += 1

    # Record cost
    _cost_tracker.record(
        CostRecord(
            provider=result.provider,
            model=result.model,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            cost_usd=result.cost_usd,
            quality_score=result.quality.score,
            latency_ms=int(result.latency_ms),
            status=result.status,
            role=result.role,
        )
    )

    if result.status in ("failed", "blocked"):
        return JSONResponse(status_code=502, content=result.response)

    return result.response


async def _forward_unknown_provider(
    body: dict,
    provider_name: str,
    role: str | None,
    target_format: str,
) -> RoutingResult:
    """Forward a request to a provider not in Daneel's config.

    Seldon may call providers Daneel doesn't know about.  We still apply
    quality gating but skip circuit breaking (no health tracking).
    Cost is tracked at $0 since we don't know the rate.
    """
    # Try to resolve from seldon_integration.extra_providers in config
    extra = _config.get("seldon_integration", {}).get("extra_providers", {})
    pconfig = extra.get(provider_name, {})

    endpoint = pconfig.get("endpoint") or body.get("endpoint")
    if not endpoint:
        logger.warning(
            "Unknown provider '%s' with no endpoint in config or request body",
            provider_name,
        )
        return RoutingResult(
            provider=provider_name,
            model=body.get("model", "unknown"),
            response={"error": {"message": f"No endpoint for unknown provider '{provider_name}'", "type": "server_error", "code": 502}},
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.0,
            quality=QualityResult(score=0.0, passed=False),
            latency_ms=0.0,
            status="failed",
            error=f"No endpoint for provider '{provider_name}'",
            role=role,
        )

    # Resolve API key
    key_env = pconfig.get("api_key_env", "")
    api_key = os.environ.get(key_env, "") if key_env else ""

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    model = body.get("model") or pconfig.get("model", "unknown")
    quality_threshold = _config.get("quality_gate", {}).get("threshold", 0.5)

    logger.warning(
        "Forwarding to unknown provider '%s' (role=%s, model=%s, endpoint=%s)",
        provider_name,
        role,
        model,
        endpoint,
    )

    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            # Strip non-standard fields before forwarding
            forward_body = {k: v for k, v in body.items() if k != "endpoint"}
            resp = await client.post(endpoint, json=forward_body, headers=headers)
        latency_ms = (time.monotonic() - start) * 1000

        if resp.status_code != 200:
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
                error=f"HTTP {resp.status_code}: {resp.text[:500]}",
                role=role,
            )

        data = resp.json()

        # Extract and quality-gate the response
        choices = data.get("choices", [])
        response_text = ""
        if choices:
            msg = choices[0].get("message", {})
            response_text = msg.get("content", "") or ""

        quality = score_response(response_text, quality_threshold)

        # Scrub credentials
        scrubbed, was_scrubbed = scrub_credentials(response_text)
        if was_scrubbed and choices and "message" in choices[0]:
            data["choices"][0]["message"]["content"] = scrubbed

        # Calculate cost from extra_providers config (or $0)
        usage = data.get("usage", {})
        tokens_in = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
        tokens_out = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)
        cost_config = pconfig.get("cost_per_million", {})
        cost = round(
            (tokens_in / 1_000_000) * cost_config.get("input", 0)
            + (tokens_out / 1_000_000) * cost_config.get("output", 0),
            8,
        )

        return RoutingResult(
            provider=provider_name,
            model=model,
            response=data,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
            quality=quality,
            latency_ms=latency_ms,
            status="success" if quality.passed else "rejected",
            role=role,
        )

    except Exception as exc:
        latency_ms = (time.monotonic() - start) * 1000
        logger.exception("Unknown provider '%s' error", provider_name)
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
            role=role,
        )


def create_app() -> FastAPI:
    """Factory function for programmatic use."""
    return app
