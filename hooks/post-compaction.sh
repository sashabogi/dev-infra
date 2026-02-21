#!/bin/bash
# Memory Rescue — extract facts/decisions/skills before compaction destroys them
#
# Event:   PostToolCall
# Purpose: Detects context compaction tool calls and dispatches the rescued
#          text to the dev-infra Memory Rescue endpoint via rescue_helper.py.
#
# Exit codes:
#   0 — always (PostToolCall hooks cannot block; we just observe)

set -euo pipefail

HOOK_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="$HOME/.dev-infra/hooks.log"
mkdir -p "$HOME/.dev-infra"

# Read the full hook payload from stdin into a variable
PAYLOAD=$(cat)

# Fast-path: extract tool_name with python3 (available on macOS + most Linux)
TOOL_NAME=$(printf '%s' "$PAYLOAD" | python3 -c \
    "import sys,json; print(json.load(sys.stdin).get('tool_name',''))" 2>/dev/null || echo "")

# Only fire on compaction / summarization events
case "$TOOL_NAME" in
    *compact*|*summariz*|*compress*|*Compact*|*Summariz*|*Compress*)
        ;;
    *)
        exit 0
        ;;
esac

# Dispatch rescue in background so we never block Claude Code
printf '%s' "$PAYLOAD" | python3 "$HOOK_DIR/rescue_helper.py" &

CONTEXT_LEN=$(printf '%s' "$PAYLOAD" | wc -c | tr -d ' ')
echo "$(date -u +%FT%TZ) [rescue] Dispatched memory rescue (tool=$TOOL_NAME payload_bytes=$CONTEXT_LEN)" >> "$LOG_FILE"

exit 0
