#!/bin/bash
# Uninstall dev-infra hooks from Claude Code
#
# Removes hook scripts from ~/.dev-infra/hooks/ and strips
# dev-infra entries from ~/.claude/hooks.json.
#
# This script is idempotent — safe to run even if not installed.
#
# Usage:
#   bash /path/to/dev-infra/hooks/uninstall.sh

set -euo pipefail

INSTALL_DIR="$HOME/.dev-infra/hooks"
CLAUDE_HOOKS_FILE="$HOME/.claude/hooks.json"

echo "Uninstalling dev-infra hooks..."
echo ""

# ── 1. Remove hook scripts ───────────────────────────────────────────
if [ -d "$INSTALL_DIR" ]; then
    rm -rf "$INSTALL_DIR"
    echo "  Removed hook scripts from $INSTALL_DIR"
else
    echo "  No hook scripts found at $INSTALL_DIR (already removed)"
fi

# ── 2. Remove dev-infra entries from hooks.json ──────────────────────
if [ -f "$CLAUDE_HOOKS_FILE" ]; then
    python3 -c "
import json

with open('$CLAUDE_HOOKS_FILE') as f:
    data = json.load(f)

hooks_before = data.get('hooks', [])
count_before = len(hooks_before)

data['hooks'] = [h for h in hooks_before if '.dev-infra' not in h.get('command', '')]
count_after = len(data['hooks'])
removed = count_before - count_after

with open('$CLAUDE_HOOKS_FILE', 'w') as f:
    json.dump(data, f, indent=2)
    f.write('\n')

print(f'  Removed {removed} dev-infra hooks from hooks.json ({count_after} hooks remaining)')
"
else
    echo "  No hooks.json found at $CLAUDE_HOOKS_FILE (nothing to clean)"
fi

echo ""
echo "Dev-infra hooks uninstalled successfully."
echo ""
echo "  Note: ~/.dev-infra/hooks.log was preserved for debugging."
echo "  To remove logs: rm -f ~/.dev-infra/hooks.log"
