#!/bin/bash
# Uninstall dev-infra hooks from Claude Code
#
# Removes hook scripts from ~/.dev-infra/hooks/ and strips
# dev-infra entries from ~/.claude/settings.json hooks object.
#
# This script is idempotent — safe to run even if not installed.
#
# Usage:
#   bash /path/to/dev-infra/hooks/uninstall.sh

set -euo pipefail

INSTALL_DIR="$HOME/.dev-infra/hooks"
CLAUDE_SETTINGS_FILE="$HOME/.claude/settings.json"

echo "Uninstalling dev-infra hooks..."
echo ""

# ── 1. Remove hook scripts ───────────────────────────────────────────
if [ -d "$INSTALL_DIR" ]; then
    rm -rf "$INSTALL_DIR"
    echo "  Removed hook scripts from $INSTALL_DIR"
else
    echo "  No hook scripts found at $INSTALL_DIR (already removed)"
fi

# ── 2. Remove dev-infra entries from settings.json hooks ─────────────
if [ -f "$CLAUDE_SETTINGS_FILE" ]; then
    python3 << 'PYTHON_SCRIPT'
import json
import os

settings_file = os.path.expanduser("~/.claude/settings.json")

with open(settings_file) as f:
    settings = json.load(f)

hooks = settings.get("hooks", {})
removed = 0

for event_name in list(hooks.keys()):
    entries = hooks[event_name]
    if not isinstance(entries, list):
        continue

    filtered = []
    for entry in entries:
        is_dev_infra = False
        for h in entry.get("hooks", []):
            if ".dev-infra" in h.get("command", ""):
                is_dev_infra = True
                break
        if is_dev_infra:
            removed += 1
        else:
            filtered.append(entry)

    if filtered:
        hooks[event_name] = filtered
    else:
        # Remove the event key entirely if no hooks remain
        del hooks[event_name]

settings["hooks"] = hooks

with open(settings_file, "w") as f:
    json.dump(settings, f, indent=2)
    f.write("\n")

print(f"  Removed {removed} dev-infra hooks from settings.json")
PYTHON_SCRIPT
else
    echo "  No settings.json found at $CLAUDE_SETTINGS_FILE (nothing to clean)"
fi

echo ""
echo "Dev-infra hooks uninstalled successfully."
echo ""
echo "  Note: ~/.dev-infra/hooks.log was preserved for debugging."
echo "  To remove logs: rm -f ~/.dev-infra/hooks.log"
