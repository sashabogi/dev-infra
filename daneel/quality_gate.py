"""Pre-flight quality scoring for cheap model outputs."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class QualityResult:
    score: float
    passed: bool
    checks: dict[str, float] = field(default_factory=dict)
    details: list[str] = field(default_factory=list)


# Fake XML tags cheap models love to hallucinate
_XML_HALLUCINATION_PATTERNS = [
    re.compile(r"<minimax[:_][\w_]+>", re.IGNORECASE),
    re.compile(r"</?invoke\b[^>]*>", re.IGNORECASE),
    re.compile(r"</?function_call\b[^>]*>", re.IGNORECASE),
    re.compile(r"</?tool_call\b[^>]*>", re.IGNORECASE),
    re.compile(r"</?tool_use\b[^>]*>", re.IGNORECASE),
    re.compile(r"</?tool_result\b[^>]*>", re.IGNORECASE),
    re.compile(r"</?assistant_response\b[^>]*>", re.IGNORECASE),
    re.compile(r"</?system_instruction\b[^>]*>", re.IGNORECASE),
    re.compile(r"</?internal_thought\b[^>]*>", re.IGNORECASE),
    re.compile(r"</?reasoning\b[^>]*>", re.IGNORECASE),
]

# Prompt injection / role confusion markers
_INJECTION_PATTERNS = [
    re.compile(r"\[SYSTEM\]", re.IGNORECASE),
    re.compile(r"\[INST\]", re.IGNORECASE),
    re.compile(r"\[/INST\]", re.IGNORECASE),
    re.compile(r"<<SYS>>", re.IGNORECASE),
    re.compile(r"<</SYS>>", re.IGNORECASE),
    re.compile(r"<\|im_start\|>", re.IGNORECASE),
    re.compile(r"<\|im_end\|>", re.IGNORECASE),
    re.compile(r"<\|system\|>", re.IGNORECASE),
    re.compile(r"<\|user\|>", re.IGNORECASE),
    re.compile(r"<\|assistant\|>", re.IGNORECASE),
    re.compile(r"Human:\s*$", re.MULTILINE),
    re.compile(r"Assistant:\s*$", re.MULTILINE),
    re.compile(r"^system\s*:", re.MULTILINE | re.IGNORECASE),
]

# Gibberish detection: long runs of repeated characters or non-ASCII noise
_GIBBERISH_PATTERN = re.compile(r"(.)\1{20,}")
_EXCESSIVE_NONASCII = re.compile(r"[^\x00-\x7F]{50,}")


def check_xml_hallucination(text: str) -> tuple[float, list[str]]:
    """Check for hallucinated XML tags. Penalty: -0.4."""
    hits = []
    for pat in _XML_HALLUCINATION_PATTERNS:
        matches = pat.findall(text)
        if matches:
            hits.extend(matches[:3])

    if hits:
        return -0.4, [f"XML hallucination detected: {', '.join(hits[:5])}"]
    return 0.0, []


def check_formatting(text: str) -> tuple[float, list[str]]:
    """Check for formatting violations. Penalty: -0.2."""
    issues = []
    penalty = 0.0

    # Completely empty or whitespace-only structured content
    stripped = text.strip()
    if not stripped:
        return -0.2, ["Empty response"]

    # Check for malformed JSON if response looks like it should be JSON
    if stripped.startswith("{") or stripped.startswith("["):
        bracket_count = stripped.count("{") - stripped.count("}")
        square_count = stripped.count("[") - stripped.count("]")
        if abs(bracket_count) > 1 or abs(square_count) > 1:
            issues.append("Unbalanced brackets in JSON-like response")
            penalty = -0.2

    return penalty, issues


def check_prompt_injection(text: str) -> tuple[float, list[str]]:
    """Check for prompt injection / role confusion. Penalty: -0.5."""
    hits = []
    for pat in _INJECTION_PATTERNS:
        matches = pat.findall(text)
        if matches:
            hits.extend(str(m).strip() for m in matches[:2])

    if hits:
        return -0.5, [f"Prompt injection markers: {', '.join(hits[:5])}"]
    return 0.0, []


def check_coherence(text: str) -> tuple[float, list[str]]:
    """Check for empty, truncated, or gibberish output. Penalty: -0.3."""
    issues = []
    penalty = 0.0

    stripped = text.strip()

    # Empty
    if not stripped:
        return -0.3, ["Empty response"]

    # Very short (likely truncated)
    if len(stripped) < 10:
        issues.append(f"Suspiciously short response ({len(stripped)} chars)")
        penalty = -0.3

    # Gibberish: repeated characters
    if _GIBBERISH_PATTERN.search(stripped):
        issues.append("Repeated character pattern detected")
        penalty = min(penalty, -0.3)

    # Excessive non-ASCII noise
    if _EXCESSIVE_NONASCII.search(stripped):
        issues.append("Excessive non-ASCII character sequence")
        penalty = min(penalty, -0.3)

    # Truncated mid-sentence (ends with common truncation artifacts)
    if stripped.endswith("...") and len(stripped) > 100:
        pass  # Ellipsis is sometimes intentional
    elif stripped[-1] not in ".!?}])\"\'\n`" and len(stripped) > 200:
        # Long response that doesn't end with sentence-ending punctuation
        issues.append("Response may be truncated (no terminal punctuation)")
        penalty = min(penalty, -0.15)

    return penalty, issues


def score_response(text: str, threshold: float = 0.5) -> QualityResult:
    """Score a response. Returns QualityResult with score 0.0-1.0."""
    score = 1.0
    all_details: list[str] = []
    checks: dict[str, float] = {}

    for name, checker in [
        ("xml_hallucination", check_xml_hallucination),
        ("formatting_violation", check_formatting),
        ("prompt_injection", check_prompt_injection),
        ("coherence", check_coherence),
    ]:
        penalty, details = checker(text)
        checks[name] = penalty
        score += penalty
        all_details.extend(details)

    score = max(0.0, min(1.0, score))

    return QualityResult(
        score=round(score, 3),
        passed=score >= threshold,
        checks=checks,
        details=all_details,
    )
