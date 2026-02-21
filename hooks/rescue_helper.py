#!/usr/bin/env python3
"""Helper script called by Claude Code hooks to trigger memory rescue.

Reads a hook payload from stdin (or a file path argument), extracts
conversation context, and sends it to the dev-infra Memory Rescue
endpoint for fact/decision/skill extraction.

Usage:
    echo '{"tool_input": {...}}' | python3 rescue_helper.py
    python3 rescue_helper.py /tmp/payload.json
"""

import json
import os
import sys
from datetime import datetime, timezone

DEV_INFRA_URL = os.environ.get("DEV_INFRA_URL", "http://localhost:8889")
LOG_FILE = os.path.expanduser("~/.dev-infra/hooks.log")
MAX_CONTEXT_CHARS = 100_000


def log(msg: str) -> None:
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    with open(LOG_FILE, "a") as f:
        f.write(f"{ts} {msg}\n")


def extract_context(payload: dict) -> str:
    """Pull the largest useful text blob from the hook payload."""
    tool_input = payload.get("tool_input", {})

    if isinstance(tool_input, dict):
        # Prefer explicit content fields, fall back to the whole dict
        context = tool_input.get("content") or tool_input.get("text") or json.dumps(tool_input)
    elif isinstance(tool_input, str):
        context = tool_input
    else:
        context = json.dumps(payload)

    # Also include tool_result if present (compaction output)
    tool_result = payload.get("tool_result", "")
    if isinstance(tool_result, str) and len(tool_result) > 100:
        context = f"{context}\n\n--- tool_result ---\n{tool_result}"

    return context[:MAX_CONTEXT_CHARS]


def main() -> None:
    # Read payload ----------------------------------------------------------
    try:
        if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
            with open(sys.argv[1]) as f:
                payload = json.load(f)
        else:
            payload = json.load(sys.stdin)
    except (json.JSONDecodeError, OSError) as exc:
        log(f"[rescue] Failed to read payload: {exc}")
        sys.exit(1)

    # Extract context -------------------------------------------------------
    context = extract_context(payload)

    if len(context) < 100:
        log(f"[rescue] Context too short ({len(context)} chars), skipping")
        return

    project = payload.get("cwd", os.environ.get("CLAUDE_PROJECT_DIR", "unknown"))
    session_id = os.environ.get("CLAUDE_SESSION_ID", payload.get("session_id", "unknown"))

    # Send to Memory Rescue -------------------------------------------------
    try:
        import httpx  # noqa: delayed import — avoid startup cost when not needed

        resp = httpx.post(
            f"{DEV_INFRA_URL}/rescue",
            json={
                "context": context,
                "project": project,
                "session_id": session_id,
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        result = resp.json()
        committed = result.get("committed", result.get("count", 0))
        log(f"[rescue] Extracted {committed} memories for project={project} session={session_id}")
    except ImportError:
        # httpx not installed — fall back to urllib
        import urllib.request

        body = json.dumps({
            "context": context,
            "project": project,
            "session_id": session_id,
        }).encode()
        req = urllib.request.Request(
            f"{DEV_INFRA_URL}/rescue",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
                committed = result.get("committed", result.get("count", 0))
                log(f"[rescue] Extracted {committed} memories for project={project} session={session_id}")
        except Exception as exc:
            log(f"[rescue] urllib fallback error: {exc}")
    except Exception as exc:
        log(f"[rescue] Error sending to rescue endpoint: {exc}")


if __name__ == "__main__":
    main()
