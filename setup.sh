#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────
# yt2notion — One-click setup script
#
# Installs all dependencies for the YouTube → Notion pipeline:
#   1. Python packages (pip)
#   2. whisper.cpp (compiled with Metal GPU on macOS)
#   3. Whisper large-v3-turbo model (~1.6 GB download)
#   4. Node.js (for yt-dlp JS challenge solving)
#   5. Claude CLI (for summarization — requires Claude subscription)
#
# Tested on: macOS with Apple Silicon (M1/M2/M3/M4)
# ─────────────────────────────────────────────────────────

echo "╔════════════════════════════════════════╗"
echo "║     yt2notion — Setup Script           ║"
echo "╚════════════════════════════════════════╝"
echo ""

# ── 0. Check prerequisites ───────────────────────────────
check_cmd() {
    if ! command -v "$1" &>/dev/null; then
        echo "❌ Required: $1 not found. $2"
        return 1
    fi
    echo "✅ $1 found"
    return 0
}

check_cmd python3 "Install Python 3.10+ from https://python.org" || exit 1
check_cmd git "Install git: xcode-select --install" || exit 1
check_cmd cmake "Install cmake: brew install cmake" || exit 1

echo ""

# ── 1. Python packages ──────────────────────────────────
echo "📦 [1/5] Installing Python packages..."
pip3 install -r requirements.txt 2>/dev/null || pip install -r requirements.txt
echo "✅ Python packages installed"
echo ""

# ── 2. whisper.cpp (Metal GPU) ───────────────────────────
WHISPER_DIR="$HOME/.local/whisper-cpp"
WHISPER_BIN="$WHISPER_DIR/whisper-cli"

if [ -f "$WHISPER_BIN" ]; then
    echo "✅ [2/5] whisper.cpp already installed at $WHISPER_BIN"
else
    echo "🔨 [2/5] Building whisper.cpp with Metal GPU support..."
    mkdir -p "$WHISPER_DIR/src"

    if [ ! -d "$WHISPER_DIR/src/.git" ]; then
        git clone https://github.com/ggerganov/whisper.cpp.git "$WHISPER_DIR/src"
    else
        cd "$WHISPER_DIR/src" && git pull
    fi

    cd "$WHISPER_DIR/src"
    cmake -B build \
        -DGGML_METAL=ON \
        -DBUILD_SHARED_LIBS=OFF \
        -DWHISPER_BUILD_EXAMPLES=ON \
        -DCMAKE_BUILD_TYPE=Release
    cmake --build build --config Release -j$(sysctl -n hw.ncpu)

    cp build/bin/whisper-cli "$WHISPER_DIR/"
    echo "✅ whisper.cpp built and installed at $WHISPER_BIN"
fi
echo ""

# ── 3. Whisper model (large-v3-turbo) ────────────────────
MODEL_DIR="$WHISPER_DIR/models"
MODEL_FILE="$MODEL_DIR/ggml-large-v3-turbo.bin"

if [ -f "$MODEL_FILE" ]; then
    echo "✅ [3/5] Whisper model already downloaded"
else
    echo "📥 [3/5] Downloading whisper large-v3-turbo model (~1.6 GB)..."
    mkdir -p "$MODEL_DIR"

    MODEL_URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin"
    if command -v curl &>/dev/null; then
        curl -L -o "$MODEL_FILE" "$MODEL_URL"
    elif command -v wget &>/dev/null; then
        wget -O "$MODEL_FILE" "$MODEL_URL"
    else
        echo "❌ Need curl or wget to download model"
        exit 1
    fi
    echo "✅ Model downloaded to $MODEL_FILE"
fi
echo ""

# ── 4. Node.js (for yt-dlp) ─────────────────────────────
NODE_DIR="$HOME/.local/node"
NODE_BIN="$NODE_DIR/bin/node"

if [ -f "$NODE_BIN" ]; then
    echo "✅ [4/5] Node.js already installed"
else
    echo "📥 [4/5] Installing Node.js..."
    mkdir -p "$NODE_DIR"

    ARCH=$(uname -m)
    if [ "$ARCH" = "arm64" ]; then
        NODE_URL="https://nodejs.org/dist/v20.11.0/node-v20.11.0-darwin-arm64.tar.xz"
    else
        NODE_URL="https://nodejs.org/dist/v20.11.0/node-v20.11.0-darwin-x64.tar.xz"
    fi

    TMPFILE=$(mktemp /tmp/node.XXXXXX.tar.xz)
    curl -L -o "$TMPFILE" "$NODE_URL"
    tar xf "$TMPFILE" -C "$NODE_DIR" --strip-components=1
    rm "$TMPFILE"
    echo "✅ Node.js installed at $NODE_BIN"
fi
echo ""

# ── 5. Claude CLI ────────────────────────────────────────
CLAUDE_BIN="$HOME/.local/bin/claude"

if [ -f "$CLAUDE_BIN" ]; then
    echo "✅ [5/5] Claude CLI already installed"
else
    echo "📥 [5/5] Installing Claude CLI..."

    # Try npm install (most reliable)
    if command -v npm &>/dev/null || [ -f "$NODE_DIR/bin/npm" ]; then
        NPM="${NODE_DIR}/bin/npm"
        [ ! -f "$NPM" ] && NPM="npm"
        mkdir -p "$HOME/.local/bin"
        $NPM install -g @anthropic-ai/claude-code 2>/dev/null && \
            echo "✅ Claude CLI installed" || \
            echo "⚠️  Claude CLI install failed — install manually: npm install -g @anthropic-ai/claude-code"
    else
        echo "⚠️  npm not found. Install Claude CLI manually:"
        echo "   npm install -g @anthropic-ai/claude-code"
    fi
fi
echo ""

# ── 6. Install pre-commit hook (secret leak prevention) ──
echo "🔒 [6/6] Installing pre-commit hook..."
HOOK_SRC="$SCRIPT_DIR/.githooks/pre-commit"
HOOK_DST="$SCRIPT_DIR/.git/hooks/pre-commit"
if [ -f "$HOOK_SRC" ] && [ -d "$SCRIPT_DIR/.git/hooks" ]; then
    cp "$HOOK_SRC" "$HOOK_DST"
    chmod +x "$HOOK_DST"
    echo "✅ Pre-commit hook installed (blocks accidental secret commits)"
else
    echo "⚠️  Could not install pre-commit hook (not a git repo?)"
fi
echo ""

# ── 7. Check .env ────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ ! -f "$SCRIPT_DIR/.env" ]; then
    echo "⚠️  No .env file found!"
    echo "   Copy and edit the example:"
    echo "   cp .env.example .env"
    echo "   Then fill in your NOTION_API_KEY and NOTION_DATABASE_ID"
    echo ""
fi

# ── Done ─────────────────────────────────────────────────
echo "╔════════════════════════════════════════╗"
echo "║          Setup Complete!               ║"
echo "╚════════════════════════════════════════╝"
echo ""
echo "Quick start:"
echo "  1. cp .env.example .env  (fill in your Notion keys)"
echo "  2. claude  (login to Claude if first time)"
echo "  3. echo 'https://youtu.be/VIDEO_ID' | python3 yt2notion.py"
echo ""
echo "Or use in Claude Code chat:"
echo '  "Save this to Notion: https://youtu.be/VIDEO_ID"'
echo ""
