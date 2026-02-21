"""Daneel FastAPI server — transparent LLM inference proxy."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from daneel.costs import CostRecord, CostTracker
from daneel.router import Router, anthropic_to_openai_messages

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

    kwargs: dict[str, Any] = {}
    for k in ("temperature", "max_tokens", "top_p", "stop"):
        if k in body:
            kwargs[k] = body[k]

    result = await _router.route_request(
        messages=messages,
        model=model,
        target_format="openai",
        **kwargs,
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
        **kwargs,
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
        )
    )

    if result.status in ("failed", "blocked"):
        return JSONResponse(status_code=502, content=result.response)

    return result.response


def create_app() -> FastAPI:
    """Factory function for programmatic use."""
    return app
