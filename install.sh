#!/bin/bash
# dev-infra installer
# Usage: git clone https://github.com/USER/dev-infra && cd dev-infra && bash install.sh
set -euo pipefail

echo "============================================"
echo "  dev-infra installer"
echo "  Daneel Proxy + Memory Rescue + Hooks"
echo "============================================"
echo ""

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_DIR="$HOME/.dev-infra"

# Step 1: Check prerequisites
echo "[1/7] Checking prerequisites..."

# Python 3.10+
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install Python 3.10+ first."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]; }; then
    echo "ERROR: Python 3.10+ required (found $PYTHON_VERSION)"
    exit 1
fi
echo "  Python $PYTHON_VERSION OK"

# pip
if ! python3 -m pip --version &>/dev/null; then
    echo "ERROR: pip not found. Install with: python3 -m ensurepip"
    exit 1
fi
echo "  pip OK"

# Claude Code (optional but recommended)
if command -v claude &>/dev/null; then
    CLAUDE_VERSION=$(claude --version 2>/dev/null || echo "unknown")
    echo "  Claude Code $CLAUDE_VERSION OK"
else
    echo "  Claude Code not found (hooks will install but won't activate until Claude Code is present)"
fi

# Step 2: Create install directory
echo ""
echo "[2/7] Creating ~/.dev-infra/ ..."
mkdir -p "$INSTALL_DIR"
mkdir -p "$INSTALL_DIR/hooks"
mkdir -p "$INSTALL_DIR/logs"

# Step 3: Install Python package
echo ""
echo "[3/7] Installing Python package..."
cd "$REPO_DIR"
python3 -m pip install -e . --quiet 2>&1 | tail -3

# Verify CLI installed
if command -v dev-infra &>/dev/null; then
    echo "  dev-infra CLI OK"
else
    # Check common pip install locations
    for candidate in "$HOME/.local/bin/dev-infra" "$HOME/Library/Python/$PYTHON_VERSION/bin/dev-infra"; do
        if [ -f "$candidate" ]; then
            echo "  dev-infra CLI installed at $candidate"
            echo "  WARNING: $candidate is not in your PATH"
            echo "  Add this to your shell profile:"
            echo "    export PATH=\"$(dirname "$candidate"):\$PATH\""
            break
        fi
    done
    if ! command -v dev-infra &>/dev/null; then
        echo "  WARNING: dev-infra CLI not found in PATH. You may need to add ~/.local/bin to your PATH"
    fi
fi

# Step 4: Copy config
echo ""
echo "[4/7] Setting up configuration..."
if [ ! -f "$INSTALL_DIR/config.yaml" ]; then
    cp "$REPO_DIR/config.example.yaml" "$INSTALL_DIR/config.yaml"
    echo "  Created $INSTALL_DIR/config.yaml"
    echo "  IMPORTANT: Edit this file to customize provider settings"
else
    echo "  Config already exists, skipping (delete $INSTALL_DIR/config.yaml to reset)"
fi

# Step 5: Set up environment
echo ""
echo "[5/7] Setting up environment..."
if [ ! -f "$INSTALL_DIR/.env" ]; then
    cp "$REPO_DIR/.env.example" "$INSTALL_DIR/.env"
    echo "  Created $INSTALL_DIR/.env"
    echo "  IMPORTANT: Add your API keys to $INSTALL_DIR/.env"
else
    echo "  .env already exists, skipping"
fi

# Step 6: Install Claude Code hooks
echo ""
echo "[6/7] Installing Claude Code hooks..."
if [ -f "$REPO_DIR/hooks/install.sh" ]; then
    bash "$REPO_DIR/hooks/install.sh"
else
    echo "  WARNING: hooks/install.sh not found, skipping hook installation"
fi

# Step 7: Install macOS launchd service (optional)
echo ""
echo "[7/7] Setting up auto-start service..."

if [ "$(uname)" = "Darwin" ]; then
    PLIST_SRC="$REPO_DIR/services/com.dev-infra.daneel.plist"
    PLIST_DST="$HOME/Library/LaunchAgents/com.dev-infra.daneel.plist"

    if [ -f "$PLIST_SRC" ]; then
        # Ensure LaunchAgents directory exists
        mkdir -p "$HOME/Library/LaunchAgents"

        # Unload existing if present
        launchctl unload "$PLIST_DST" 2>/dev/null || true

        # Template the plist with actual paths
        sed "s|__HOME__|$HOME|g; s|__REPO__|$REPO_DIR|g" "$PLIST_SRC" > "$PLIST_DST"

        echo "  Installed launchd service"
        echo "  To auto-start Daneel on login: launchctl load $PLIST_DST"
        echo "  To start now: dev-infra start"
    else
        echo "  WARNING: services/com.dev-infra.daneel.plist not found, skipping"
    fi
else
    echo "  Skipping launchd setup (macOS only)"
    echo "  On Linux, create a systemd unit or use a process manager of your choice"
fi

echo ""
echo "============================================"
echo "  Installation complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Add API keys:  vim $INSTALL_DIR/.env"
echo "  2. Edit config:   vim $INSTALL_DIR/config.yaml"
echo "  3. Start Daneel:   dev-infra start"
echo "  4. Check status:  dev-infra status"
echo ""
echo "Daneel proxy will listen on http://localhost:8889"
echo "Point your apps at it: LLM_BASE_URL=http://localhost:8889/v1"
echo ""
