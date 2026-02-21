#!/bin/bash
# dev-infra uninstaller
set -euo pipefail

echo "Uninstalling dev-infra..."

# Stop Daneel if running
if command -v dev-infra &>/dev/null; then
    dev-infra stop 2>/dev/null || true
fi

# Unload launchd service (macOS)
if [ "$(uname)" = "Darwin" ]; then
    PLIST="$HOME/Library/LaunchAgents/com.dev-infra.daneel.plist"
    if [ -f "$PLIST" ]; then
        launchctl unload "$PLIST" 2>/dev/null || true
        rm -f "$PLIST"
        echo "  Removed launchd service"
    fi
fi

# Uninstall hooks
if [ -f "$HOME/.dev-infra/hooks/uninstall.sh" ]; then
    bash "$HOME/.dev-infra/hooks/uninstall.sh"
fi
# Also try repo-local hooks uninstaller
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$REPO_DIR/hooks/uninstall.sh" ]; then
    bash "$REPO_DIR/hooks/uninstall.sh"
fi

# Uninstall Python package
python3 -m pip uninstall dev-infra -y 2>/dev/null || true
echo "  Removed Python package"

# Ask about data
echo ""
echo "Remove data files? This includes:"
echo "  - $HOME/.dev-infra/memory.db (rescued memories)"
echo "  - $HOME/.dev-infra/costs.db (cost tracking)"
echo "  - $HOME/.dev-infra/config.yaml"
echo "  - $HOME/.dev-infra/.env (API keys)"
echo "  - $HOME/.dev-infra/logs/"
read -p "Delete all data? [y/N] " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf "$HOME/.dev-infra"
    echo "  Removed all data"
else
    echo "  Data preserved at $HOME/.dev-infra/"
fi

echo ""
echo "dev-infra uninstalled."
