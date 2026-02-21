#!/bin/bash
# Session context loader — retrieves rescued memories relevant to the current project
#
# This script is NOT wired as a Claude Code hook event (there is no SessionStart
# event). Instead it can be:
#   - Called manually:  bash ~/.dev-infra/hooks/session-start.sh
#   - Sourced from a shell profile for interactive use
#   - Invoked by a wrapper that starts Claude Code sessions
#
# It queries the dev-infra memory store for recent facts/decisions/skills
# related to the current project and prints them to stdout.

set -euo pipefail

DEV_INFRA_URL="${DEV_INFRA_URL:-http://localhost:8889}"
PROJECT="${CLAUDE_PROJECT_DIR:-$(pwd)}"
PROJECT_NAME="$(basename "$PROJECT")"
LIMIT="${1:-5}"

# Attempt to reach the memory search endpoint
RESPONSE=$(curl -sf --max-time 5 \
    "$DEV_INFRA_URL/memories/search?project=$PROJECT_NAME&limit=$LIMIT" 2>/dev/null) || {
    # Server not running or endpoint unavailable — silently exit
    exit 0
}

# Check if we got any memories back
MEMORY_COUNT=$(printf '%s' "$RESPONSE" | python3 -c "
import sys, json
data = json.load(sys.stdin)
memories = data if isinstance(data, list) else data.get('memories', [])
print(len(memories))
" 2>/dev/null || echo "0")

if [ "$MEMORY_COUNT" = "0" ] || [ -z "$MEMORY_COUNT" ]; then
    exit 0
fi

# Pretty-print the memories
echo ""
echo "=== Rescued Memories for $PROJECT_NAME ($MEMORY_COUNT found) ==="
printf '%s' "$RESPONSE" | python3 -c "
import sys, json

data = json.load(sys.stdin)
memories = data if isinstance(data, list) else data.get('memories', [])

for m in memories:
    category = m.get('category', '?').upper()
    content = m.get('content', m.get('text', '')).strip()
    # Truncate long entries for display
    if len(content) > 120:
        content = content[:117] + '...'
    print(f'  [{category}] {content}')
" 2>/dev/null || true
echo "=== End Memories ==="
echo ""

exit 0
