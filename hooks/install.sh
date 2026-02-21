#!/bin/bash
# Install dev-infra hooks into Claude Code
#
# This script is idempotent — safe to run multiple times.
# It copies hook scripts to ~/.dev-infra/hooks/ and merges
# the hook configuration into ~/.claude/settings.json under the
# "hooks" key using the correct Claude Code format:
#
#   hooks: {
#     PreToolUse: [ { matcher: "Bash", hooks: [{ type: "command", command: "..." }] } ],
#     PostToolUse: [ { matcher: "Bash", hooks: [{ type: "command", command: "..." }] } ],
#     Stop: [ { hooks: [{ type: "command", command: "..." }] } ]
#   }
#
# Usage:
#   bash /path/to/dev-infra/hooks/install.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_DIR="$HOME/.dev-infra/hooks"
CLAUDE_SETTINGS_FILE="$HOME/.claude/settings.json"

echo "Installing dev-infra hooks..."
echo ""

# ── 1. Create directories ────────────────────────────────────────────
mkdir -p "$INSTALL_DIR"
mkdir -p "$HOME/.claude"

# ── 2. Copy hook scripts and helper ──────────────────────────────────
SCRIPTS=(
    "post-compaction.sh"
    "session-start.sh"
    "session-end.sh"
    "pre-tool-safety.sh"
    "rescue_helper.py"
)

for script in "${SCRIPTS[@]}"; do
    if [ ! -f "$SCRIPT_DIR/$script" ]; then
        echo "  WARNING: $script not found in $SCRIPT_DIR, skipping"
        continue
    fi
    cp "$SCRIPT_DIR/$script" "$INSTALL_DIR/$script"
    echo "  Copied $script"
done

# ── 3. Make shell scripts executable ─────────────────────────────────
chmod +x "$INSTALL_DIR"/*.sh 2>/dev/null || true

# ── 4. Merge hooks into ~/.claude/settings.json ─────────────────────
# Uses the correct Claude Code hook format with PreToolUse/PostToolUse/Stop
# events. Preserves all existing hooks (RTK, etc.) by only touching entries
# whose command contains ".dev-infra".

python3 << 'PYTHON_SCRIPT'
import json
import os
import sys

settings_file = os.path.expanduser("~/.claude/settings.json")
install_dir = os.path.expanduser("~/.dev-infra/hooks")

# Load existing settings
if os.path.exists(settings_file):
    with open(settings_file) as f:
        settings = json.load(f)
else:
    settings = {}

# Ensure hooks object exists
if "hooks" not in settings:
    settings["hooks"] = {}

hooks = settings["hooks"]

# Dev-infra hooks to install
DEV_INFRA_HOOKS = {
    "PreToolUse": {
        "matcher": "Bash",
        "hooks": [
            {
                "type": "command",
                "command": f"bash {install_dir}/pre-tool-safety.sh"
            }
        ]
    },
    "PostToolUse": {
        "matcher": "Bash",
        "hooks": [
            {
                "type": "command",
                "command": f"bash {install_dir}/post-compaction.sh"
            }
        ]
    },
    "Stop": {
        "hooks": [
            {
                "type": "command",
                "command": f"bash {install_dir}/session-end.sh"
            }
        ]
    }
}

installed_count = 0

for event_name, new_entry in DEV_INFRA_HOOKS.items():
    # Ensure the event array exists
    if event_name not in hooks:
        hooks[event_name] = []

    event_list = hooks[event_name]

    # Remove any existing dev-infra entries (idempotent)
    filtered = []
    for entry in event_list:
        is_dev_infra = False
        for h in entry.get("hooks", []):
            if ".dev-infra" in h.get("command", ""):
                is_dev_infra = True
                break
        if not is_dev_infra:
            filtered.append(entry)

    # Append the new dev-infra entry
    filtered.append(new_entry)
    hooks[event_name] = filtered
    installed_count += 1

settings["hooks"] = hooks

with open(settings_file, "w") as f:
    json.dump(settings, f, indent=2)
    f.write("\n")

print(f"  Merged {installed_count} dev-infra hooks into settings.json")
PYTHON_SCRIPT

# ── 5. Verify installation ───────────────────────────────────────────
echo ""
INSTALLED_COUNT=$(ls "$INSTALL_DIR"/*.sh 2>/dev/null | wc -l | tr -d ' ')
HOOK_COUNT=$(python3 -c "
import json, os
with open(os.path.expanduser('~/.claude/settings.json')) as f:
    d = json.load(f)
hooks = d.get('hooks', {})
count = 0
for event_name, entries in hooks.items():
    if not isinstance(entries, list):
        continue
    for entry in entries:
        for h in entry.get('hooks', []):
            if '.dev-infra' in h.get('command', ''):
                count += 1
print(count)
" 2>/dev/null || echo "?")

echo "Installation complete!"
echo ""
echo "  Scripts installed: $INSTALL_DIR/ ($INSTALLED_COUNT shell scripts)"
echo "  Hooks configured:  $CLAUDE_SETTINGS_FILE ($HOOK_COUNT dev-infra hooks)"
echo ""
echo "  Active hooks:"
echo "    PreToolUse  -> Safety gate (blocks destructive commands)"
echo "    PostToolUse -> Memory rescue (extracts context on compaction)"
echo "    Stop        -> Session checkpoint (saves context on exit)"
echo ""
echo "  Manual tools:"
echo "    bash $INSTALL_DIR/session-start.sh   # Load rescued memories"
echo ""
echo "  To uninstall:"
echo "    bash $SCRIPT_DIR/uninstall.sh"
