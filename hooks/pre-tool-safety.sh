#!/bin/bash
# Safety gate — blocks destructive shell commands before execution
#
# Event:   PreToolUse (matcher: Bash)
# Input:   {"tool_name": "Bash", "tool_input": {"command": "..."}}
# Purpose: Inspects Bash tool calls for dangerous patterns (recursive
#          deletes on system dirs, fork bombs, pipe-to-shell, etc.)
#          and blocks them with exit code 2.
#
# Exit codes:
#   0 — allow the tool call
#   2 — BLOCK the tool call (Claude Code will not execute it)

set -euo pipefail

LOG_FILE="$HOME/.dev-infra/hooks.log"

# Read the full hook payload from stdin
PAYLOAD=$(cat)

# Extract tool name — only inspect Bash/bash calls
TOOL_NAME=$(printf '%s' "$PAYLOAD" | python3 -c \
    "import sys,json; print(json.load(sys.stdin).get('tool_name',''))" 2>/dev/null || echo "")

case "$TOOL_NAME" in
    Bash|bash) ;;
    *) exit 0 ;;
esac

# Extract the command string
COMMAND=$(printf '%s' "$PAYLOAD" | python3 -c "
import sys, json
d = json.load(sys.stdin)
ti = d.get('tool_input', {})
if isinstance(ti, dict):
    print(ti.get('command', ''))
else:
    print(str(ti))
" 2>/dev/null || echo "")

# Nothing to check
if [ -z "$COMMAND" ]; then
    exit 0
fi

# --- Dangerous pattern checks ---
# Each pattern is checked with case-insensitive fixed-string grep.
# If ANY pattern matches, the command is blocked.

BLOCKED=""
MATCHED_PATTERN=""

check_pattern() {
    if printf '%s' "$COMMAND" | grep -qiF "$1"; then
        BLOCKED="yes"
        MATCHED_PATTERN="$1"
    fi
}

# Catastrophic filesystem destruction
check_pattern "rm -rf /"
check_pattern "rm -rf ~"
check_pattern 'rm -rf $HOME'
check_pattern 'rm -rf ${HOME'

# Low-level disk operations
check_pattern "dd if="
check_pattern "mkfs."

# Fork bomb
check_pattern ":(){ :|:& };:"

# Write to raw block devices
check_pattern "> /dev/sd"
check_pattern "> /dev/nvme"
check_pattern "> /dev/disk"

# Recursive permission nuke
check_pattern "chmod -R 777 /"
check_pattern "chown -R"

# Pipe-to-shell (remote code execution)
# Only block when piped directly — standalone curl/wget are fine
if printf '%s' "$COMMAND" | grep -qiE 'curl\s+.*\|\s*(ba)?sh'; then
    BLOCKED="yes"
    MATCHED_PATTERN="curl | sh"
fi
if printf '%s' "$COMMAND" | grep -qiE 'wget\s+.*\|\s*(ba)?sh'; then
    BLOCKED="yes"
    MATCHED_PATTERN="wget | sh"
fi

# --- Act on result ---
if [ -n "$BLOCKED" ]; then
    mkdir -p "$HOME/.dev-infra"
    TRUNCATED_CMD=$(printf '%s' "$COMMAND" | head -c 500)
    echo "$(date -u +%FT%TZ) [safety] BLOCKED command (matched: $MATCHED_PATTERN): $TRUNCATED_CMD" >> "$LOG_FILE"
    echo "BLOCKED by dev-infra safety gate: destructive command detected ($MATCHED_PATTERN)" >&2
    exit 2
fi

exit 0
