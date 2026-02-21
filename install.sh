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

# Python 3.10+ — try versioned binaries first, then fall back to python3
PYTHON=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        major=$(echo "$ver" | cut -d. -f1)
        minor=$(echo "$ver" | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
            PYTHON="$candidate"
            PYTHON_VERSION="$ver"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "ERROR: Python 3.10+ required. Found only $(python3 --version 2>/dev/null || echo 'no python3')."
    echo "  Install with: brew install python@3.12"
    exit 1
fi
echo "  Python $PYTHON_VERSION ($PYTHON) OK"

# pip
if ! $PYTHON -m pip --version &>/dev/null; then
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

# Step 3: Create venv and install Python package
echo ""
echo "[3/7] Installing Python package..."
VENV_DIR="$INSTALL_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "  Creating virtual environment..."
    $PYTHON -m venv "$VENV_DIR"
fi
VENV_PYTHON="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

cd "$REPO_DIR"
"$VENV_PIP" install -e . --quiet 2>&1 | tail -3

# Create a wrapper script in ~/.local/bin so `dev-infra` is on PATH
WRAPPER_DIR="$HOME/.local/bin"
mkdir -p "$WRAPPER_DIR"
cat > "$WRAPPER_DIR/dev-infra" << WRAPPER
#!/bin/bash
exec "$VENV_DIR/bin/dev-infra" "\$@"
WRAPPER
chmod +x "$WRAPPER_DIR/dev-infra"

if command -v dev-infra &>/dev/null; then
    echo "  dev-infra CLI OK"
else
    echo "  dev-infra CLI installed at $WRAPPER_DIR/dev-infra"
    echo "  WARNING: $WRAPPER_DIR is not in your PATH"
    echo "  Add this to your shell profile:"
    echo "    export PATH=\"$WRAPPER_DIR:\$PATH\""
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
