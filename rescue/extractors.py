"""Three parallel extractors: Fact, Decision, Skill."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx

from rescue.store import Memory

logger = logging.getLogger("rescue.extractors")

DANEEL_URL = "http://localhost:8889/v1/chat/completions"


@dataclass
class ExtractionConfig:
    model: str = "llama-3.3-70b"
    max_tokens: int = 4096
    temperature: float = 0.1


FACT_PROMPT = """You are a memory rescue agent. Your job is to extract FACTUAL INFORMATION from the conversation context that would be impossible to reconstruct from a summary.

Extract concrete data points:
- Names of people, projects, files, services, databases
- Dates, deadlines, version numbers, prices, quantities
- URLs, endpoints, configuration values
- Database names, table names, column names
- Specific error messages, stack traces, version pins
- API keys format hints (NOT actual keys), environment variable names

For each fact, rate its importance 1-10:
- 10: Critical configuration that would break things if lost (DB names, API endpoints)
- 8-9: Important reference data (file paths, version requirements)
- 6-7: Useful context (names, dates, preferences)
- 1-5: Nice to know but reconstructable

Return ONLY a JSON array. No other text. Each element:
{
  "text": "the factual information",
  "importance": 8,
  "subcategory": "config"
}

Valid subcategories: name, date, number, url, config, credential, spec, reference

Context to extract from:
"""

DECISION_PROMPT = """You are a memory rescue agent. Your job is to extract DECISIONS and their RATIONALE from the conversation context.

Extract:
- What was chosen AND what was rejected (with reasons)
- Trade-offs that were explicitly considered
- Constraints that drove the decision
- "We tried X but it didn't work because Y" patterns
- Architecture choices and why alternatives were ruled out
- Priority decisions (what comes first and why)

For each decision, rate its importance 1-10:
- 10: Fundamental architecture decision that affects everything
- 8-9: Important choice with significant trade-offs
- 6-7: Meaningful preference with reasoning
- 1-5: Minor choice

Return ONLY a JSON array. No other text. Each element:
{
  "text": "description of the decision, what was chosen, what was rejected, and why",
  "importance": 8,
  "subcategory": "architecture"
}

Valid subcategories: architecture, tool_choice, strategy, tradeoff, rejection, priority

Context to extract from:
"""

SKILL_PROMPT = """You are a memory rescue agent. Your job is to extract PROCEDURAL KNOWLEDGE from the conversation context.

Extract:
- Step-by-step procedures that worked
- Debugging techniques that solved problems
- Workarounds for specific issues
- Code patterns that proved effective
- Anti-patterns to avoid (and why they failed)
- Performance optimizations with measurable results
- "Gotchas" — non-obvious things that wasted time

For each skill, rate its importance 1-10:
- 10: Critical procedure that prevents data loss or major bugs
- 8-9: Debugging technique for hard-to-diagnose issues
- 6-7: Useful pattern or optimization
- 1-5: Minor tip

Return ONLY a JSON array. No other text. Each element:
{
  "text": "description of the procedural knowledge",
  "importance": 8,
  "subcategory": "how_to"
}

Valid subcategories: how_to, debug_technique, workaround, pattern, anti_pattern, optimization, gotcha

Context to extract from:
"""


async def _call_daneel(
    prompt: str,
    context: str,
    config: ExtractionConfig,
) -> list[dict[str, Any]]:
    """Send extraction request through Daneel."""
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": context},
    ]

    body = {
        "model": config.model,
        "messages": messages,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            resp = await client.post(DANEEL_URL, json=body)
            if resp.status_code != 200:
                logger.error(
                    "Daneel returned %d: %s", resp.status_code, resp.text[:500]
                )
                return []

            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                return []

            text = choices[0].get("message", {}).get("content", "")
            return _parse_json_array(text)

        except httpx.TimeoutException:
            logger.error("Daneel request timed out")
            return []
        except Exception:
            logger.exception("Daneel extraction call failed")
            return []


def _parse_json_array(text: str) -> list[dict[str, Any]]:
    """Parse JSON array from model output, handling common issues."""
    text = text.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last fence lines
        start = 1
        end = len(lines)
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip().startswith("```"):
                end = i
                break
        text = "\n".join(lines[start:end]).strip()

    # Find the JSON array boundaries
    start_idx = text.find("[")
    end_idx = text.rfind("]")
    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        logger.warning("No JSON array found in extraction response")
        return []

    json_str = text[start_idx : end_idx + 1]

    try:
        result = json.loads(json_str)
        if isinstance(result, list):
            return result
        return []
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse extraction JSON: %s", e)
        return []


class FactExtractor:
    def __init__(self, config: ExtractionConfig | None = None):
        self._config = config or ExtractionConfig()

    async def extract(
        self,
        context: str,
        project: str | None = None,
        session_id: str | None = None,
    ) -> list[Memory]:
        raw = await _call_daneel(FACT_PROMPT, context, self._config)
        memories = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            text = item.get("text", "").strip()
            if not text:
                continue
            importance = int(item.get("importance", 5))
            subcategory = item.get("subcategory", "reference")
            memories.append(
                Memory(
                    category="fact",
                    subcategory=subcategory,
                    content=text,
                    importance=importance,
                    project=project,
                    session_id=session_id,
                )
            )
        return memories


class DecisionExtractor:
    def __init__(self, config: ExtractionConfig | None = None):
        self._config = config or ExtractionConfig()

    async def extract(
        self,
        context: str,
        project: str | None = None,
        session_id: str | None = None,
    ) -> list[Memory]:
        raw = await _call_daneel(DECISION_PROMPT, context, self._config)
        memories = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            text = item.get("text", "").strip()
            if not text:
                continue
            importance = int(item.get("importance", 5))
            subcategory = item.get("subcategory", "strategy")
            memories.append(
                Memory(
                    category="decision",
                    subcategory=subcategory,
                    content=text,
                    importance=importance,
                    project=project,
                    session_id=session_id,
                )
            )
        return memories


class SkillExtractor:
    def __init__(self, config: ExtractionConfig | None = None):
        self._config = config or ExtractionConfig()

    async def extract(
        self,
        context: str,
        project: str | None = None,
        session_id: str | None = None,
    ) -> list[Memory]:
        raw = await _call_daneel(SKILL_PROMPT, context, self._config)
        memories = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            text = item.get("text", "").strip()
            if not text:
                continue
            importance = int(item.get("importance", 5))
            subcategory = item.get("subcategory", "how_to")
            memories.append(
                Memory(
                    category="skill",
                    subcategory=subcategory,
                    content=text,
                    importance=importance,
                    project=project,
                    session_id=session_id,
                )
            )
        return memories
