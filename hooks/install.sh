#!/bin/bash
# Install dev-infra hooks into Claude Code
#
# This script is idempotent — safe to run multiple times.
# It copies hook scripts to ~/.dev-infra/hooks/ and merges
# the hook configuration into ~/.claude/hooks.json.
#
# Usage:
#   bash /path/to/dev-infra/hooks/install.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_DIR="$HOME/.dev-infra/hooks"
CLAUDE_HOOKS_DIR="$HOME/.claude"
CLAUDE_HOOKS_FILE="$CLAUDE_HOOKS_DIR/hooks.json"

echo "Installing dev-infra hooks..."
echo ""

# ── 1. Create directories ────────────────────────────────────────────
mkdir -p "$INSTALL_DIR"
mkdir -p "$CLAUDE_HOOKS_DIR"

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

# ── 4. Merge hooks.json into Claude Code configuration ───────────────
if [ ! -f "$SCRIPT_DIR/hooks.json" ]; then
    echo "ERROR: hooks.json not found in $SCRIPT_DIR"
    exit 1
fi

if [ -f "$CLAUDE_HOOKS_FILE" ]; then
    echo ""
    echo "  Existing hooks.json found — merging dev-infra hooks..."
    python3 -c "
import json

with open('$CLAUDE_HOOKS_FILE') as f:
    existing = json.load(f)

with open('$SCRIPT_DIR/hooks.json') as f:
    new_hooks = json.load(f)

# Remove any previous dev-infra hooks (identified by .dev-infra in command)
existing_hooks = existing.get('hooks', [])
filtered = [h for h in existing_hooks if '.dev-infra' not in h.get('command', '')]

# Append the new dev-infra hooks
new_entries = new_hooks.get('hooks', [])
filtered.extend(new_entries)
existing['hooks'] = filtered

with open('$CLAUDE_HOOKS_FILE', 'w') as f:
    json.dump(existing, f, indent=2)
    f.write('\n')

print(f'  Merged {len(new_entries)} dev-infra hooks ({len(filtered)} total hooks)')
"
else
    echo ""
    echo "  Creating new hooks.json..."
    python3 -c "
import json

with open('$SCRIPT_DIR/hooks.json') as f:
    hooks = json.load(f)

with open('$CLAUDE_HOOKS_FILE', 'w') as f:
    json.dump(hooks, f, indent=2)
    f.write('\n')

print(f'  Created hooks.json with {len(hooks.get(\"hooks\", []))} hooks')
"
fi

# ── 5. Verify installation ───────────────────────────────────────────
echo ""
INSTALLED_COUNT=$(ls "$INSTALL_DIR"/*.sh 2>/dev/null | wc -l | tr -d ' ')
HOOK_COUNT=$(python3 -c "
import json
with open('$CLAUDE_HOOKS_FILE') as f:
    d = json.load(f)
infra = [h for h in d.get('hooks', []) if '.dev-infra' in h.get('command', '')]
print(len(infra))
" 2>/dev/null || echo "?")

echo "Installation complete!"
echo ""
echo "  Scripts installed: $INSTALL_DIR/ ($INSTALLED_COUNT shell scripts)"
echo "  Hooks configured:  $CLAUDE_HOOKS_FILE ($HOOK_COUNT dev-infra hooks)"
echo ""
echo "  Active hooks:"
echo "    PreToolCall  -> Safety gate (blocks destructive commands)"
echo "    PostToolCall -> Memory rescue (extracts context on compaction)"
echo "    Stop         -> Session checkpoint (saves context on exit)"
echo ""
echo "  Manual tools:"
echo "    bash $INSTALL_DIR/session-start.sh   # Load rescued memories"
echo ""
echo "  To uninstall:"
echo "    bash $SCRIPT_DIR/uninstall.sh"
