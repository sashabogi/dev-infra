#!/bin/bash
# Auto-checkpoint on session end — extracts remaining context before exit
#
# Event:   Stop
# Purpose: When a Claude Code session ends, the hook payload may contain
#          a final summary or conversation state. We send it through
#          Memory Rescue to capture any last facts/decisions/skills.
#
# Exit codes:
#   0 — always

set -euo pipefail

HOOK_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="$HOME/.dev-infra/hooks.log"
mkdir -p "$HOME/.dev-infra"

# Read the full hook payload from stdin
PAYLOAD=$(cat)

# Guard: if payload is empty or trivially small, skip
PAYLOAD_LEN=$(printf '%s' "$PAYLOAD" | wc -c | tr -d ' ')
if [ "$PAYLOAD_LEN" -lt 50 ]; then
    exit 0
fi

# Extract project and session info for logging
PROJECT=$(printf '%s' "$PAYLOAD" | python3 -c "
import sys, json, os
d = json.load(sys.stdin)
print(d.get('cwd', os.environ.get('CLAUDE_PROJECT_DIR', os.getcwd())))
" 2>/dev/null || echo "unknown")

SESSION_ID="${CLAUDE_SESSION_ID:-unknown}"

echo "$(date -u +%FT%TZ) [session-end] Session $SESSION_ID ended for project=$(basename "$PROJECT")" >> "$LOG_FILE"

# Check if there is meaningful content worth rescuing
HAS_CONTENT=$(printf '%s' "$PAYLOAD" | python3 -c "
import sys, json
d = json.load(sys.stdin)
# Look for summary, message, stop_reason, conversation content
content = d.get('summary', '') or d.get('message', '') or d.get('stop_reason', '')
tool_input = d.get('tool_input', {})
if isinstance(tool_input, dict):
    content = content or tool_input.get('content', '') or tool_input.get('text', '')
elif isinstance(tool_input, str):
    content = content or tool_input
print('yes' if len(str(content)) > 50 else 'no')
" 2>/dev/null || echo "no")

if [ "$HAS_CONTENT" = "yes" ]; then
    # Dispatch rescue in background — don't hold up the session exit
    printf '%s' "$PAYLOAD" | python3 "$HOOK_DIR/rescue_helper.py" &
    echo "$(date -u +%FT%TZ) [session-end] Dispatched final memory rescue (payload_bytes=$PAYLOAD_LEN)" >> "$LOG_FILE"
else
    echo "$(date -u +%FT%TZ) [session-end] No meaningful content to rescue" >> "$LOG_FILE"
fi

exit 0
