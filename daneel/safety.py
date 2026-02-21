"""Safety layer: dangerous command blocking and credential scrubbing."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class SafetyResult:
    safe: bool
    blocked_patterns: list[str] = field(default_factory=list)
    scrubbed: bool = False


# Dangerous shell commands that should never appear in prompts
_DANGEROUS_COMMANDS = [
    re.compile(r"\brm\s+-rf\s+/", re.IGNORECASE),
    re.compile(r"\brm\s+-rf\s+~", re.IGNORECASE),
    re.compile(r"\brm\s+-rf\s+\$HOME", re.IGNORECASE),
    re.compile(r"\bdd\s+if=", re.IGNORECASE),
    re.compile(r"\bmkfs\b", re.IGNORECASE),
    re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:", re.IGNORECASE),  # fork bomb
    re.compile(r"\bcurl\b[^|]*\|\s*(ba)?sh\b", re.IGNORECASE),
    re.compile(r"\bwget\b[^|]*\|\s*(ba)?sh\b", re.IGNORECASE),
    re.compile(r"\bcurl\b[^|]*\|\s*sudo\b", re.IGNORECASE),
    re.compile(r"\bwget\b[^|]*\|\s*sudo\b", re.IGNORECASE),
    re.compile(r"\b>\s*/dev/sda\b", re.IGNORECASE),
    re.compile(r"\bchmod\s+-R\s+777\s+/\s*$", re.IGNORECASE),
    re.compile(r"\bchown\s+-R\b.*\s+/\s*$", re.IGNORECASE),
]

# Credential patterns to scrub from responses
_CREDENTIAL_PATTERNS = [
    # API keys with common prefixes
    (re.compile(r"\bsk-[a-zA-Z0-9_-]{20,}\b"), "[REDACTED_API_KEY]"),
    (re.compile(r"\bsk_live_[a-zA-Z0-9_-]{20,}\b"), "[REDACTED_API_KEY]"),
    (re.compile(r"\bsk_test_[a-zA-Z0-9_-]{20,}\b"), "[REDACTED_API_KEY]"),
    (re.compile(r"\bkey-[a-zA-Z0-9_-]{20,}\b"), "[REDACTED_API_KEY]"),
    (re.compile(r"\bpk_live_[a-zA-Z0-9_-]{20,}\b"), "[REDACTED_API_KEY]"),
    (re.compile(r"\bpk_test_[a-zA-Z0-9_-]{20,}\b"), "[REDACTED_API_KEY]"),
    # Bearer tokens
    (re.compile(r"Bearer\s+[a-zA-Z0-9_.\-]{20,}"), "Bearer [REDACTED_TOKEN]"),
    # Generic long hex/base64 strings that look like keys (after = sign)
    (
        re.compile(r'(?:api[_-]?key|token|secret|password|auth)\s*[=:]\s*["\']?([a-zA-Z0-9_.\-/+]{32,})["\']?', re.IGNORECASE),
        lambda m: m.group(0).replace(m.group(1), "[REDACTED]"),
    ),
    # AWS-style keys
    (re.compile(r"\bAKIA[A-Z0-9]{16}\b"), "[REDACTED_AWS_KEY]"),
    # GitHub tokens
    (re.compile(r"\bghp_[a-zA-Z0-9]{36}\b"), "[REDACTED_GH_TOKEN]"),
    (re.compile(r"\bgho_[a-zA-Z0-9]{36}\b"), "[REDACTED_GH_TOKEN]"),
    (re.compile(r"\bghu_[a-zA-Z0-9]{36}\b"), "[REDACTED_GH_TOKEN]"),
    (re.compile(r"\bghs_[a-zA-Z0-9]{36}\b"), "[REDACTED_GH_TOKEN]"),
    # Supabase keys
    (re.compile(r"\beyJ[a-zA-Z0-9_-]{100,}\b"), "[REDACTED_JWT]"),
]


def check_safety(messages: list[dict]) -> SafetyResult:
    """Scan messages for dangerous command patterns."""
    blocked = []

    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            # Anthropic format: content is array of blocks
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            content = "\n".join(text_parts)

        for pattern in _DANGEROUS_COMMANDS:
            matches = pattern.findall(content)
            if matches:
                blocked.append(pattern.pattern)

    return SafetyResult(safe=len(blocked) == 0, blocked_patterns=blocked)


def scrub_credentials(text: str) -> tuple[str, bool]:
    """Remove credential patterns from response text. Returns (scrubbed_text, was_scrubbed)."""
    scrubbed = False
    result = text

    for pattern, replacement in _CREDENTIAL_PATTERNS:
        if callable(replacement):
            new_result = pattern.sub(replacement, result)
        else:
            new_result = pattern.sub(replacement, result)

        if new_result != result:
            scrubbed = True
            result = new_result

    return result, scrubbed
