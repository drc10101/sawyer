#!/usr/bin/env bash
# Sawyer — Distributed MoE Inference Network
# The load is split. Friends help.
#
# Usage:
#   curl -fsSL https://sawyer.infill.systems/install.sh | bash
#
# Or:
#   ./install_sawyer.sh
#
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

REPO="drc10101/sawyer"
PKG="sawyer-core"
VERSION="0.6.0"
FAST_LLAMA_TAG="sawyer-fast-llama-v0.6.0"
FAST_LLAMA_REPO="drc10101/llama.cpp"
BIN_DIR="${HOME}/.sawyer/bin"

info()  { echo -e "  ${CYAN}${1}${NC}"; }
ok()    { echo -e "  ${GREEN}${1}${NC}"; }
err()   { echo -e "  ${RED}${1}${NC}"; }

# ── Banner ──
echo ""
echo -e "  ${BOLD}${CYAN}Sawyer${NC} ${BOLD}— Distributed MoE Inference Network${NC}"
echo -e "  ${NC}The load is split. Friends help.${NC}"
echo ""

# ── Check Python ──
PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        ver=$("$cmd" --version 2>&1 || true)
        if [[ "$ver" =~ 3\.([0-9]+) ]]; then
            minor=${BASH_REMATCH[1]}
            if [ "$minor" -ge 11 ]; then
                PYTHON="$cmd"
                break
            fi
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    err "Python 3.11+ not found."
    echo "  Install Python: https://www.python.org/downloads/"
    echo "  On Ubuntu/Debian: sudo apt install python3.12 python3.12-venv"
    exit 1
fi

ok "Using $($PYTHON --version 2>&1)"

# ── Install sawyer-core via pip ──
info "Installing ${PKG} v${VERSION}..."
$PYTHON -m pip install --upgrade "$PKG" 2>&1 || {
    err "pip install failed. Try:"
    echo "  $PYTHON -m pip install --user sawyer-core"
    exit 1
}
ok "${PKG} installed"

# ── Download Sawyer Fast Llama binary ──
detect_platform() {
    local os kernel arch
    os="$(uname -s 2>/dev/null || echo unknown)"
    kernel="$(uname -m 2>/dev/null || echo unknown)"

    case "$os" in
        Linux)
            case "$kernel" in
                x86_64|amd64) echo "linux-x64" ;;
                aarch64|arm64) echo "linux-arm64" ;;
                *)             echo "linux-${kernel}" ;;
            esac
            ;;
        Darwin)
            case "$kernel" in
                x86_64|amd64) echo "macos-x64" ;;
                arm64)        echo "macos-arm64" ;;
                *)             echo "macos-${kernel}" ;;
            esac
            ;;
        *)
            echo "unsupported"
            ;;
    esac
}

PLATFORM=$(detect_platform)

if [ "$PLATFORM" = "unsupported" ]; then
    info "Sawyer Fast Llama: skipping binary download (unsupported platform)"
    info "sawyer bench will use system llama-bench if available"
else
    BINARY_NAME="sawyer-fast-llama-${PLATFORM}"
    DEST="${BIN_DIR}/${BINARY_NAME}"

    if [ -f "$DEST" ]; then
        ok "Sawyer Fast Llama already cached at ${DEST}"
    else
        info "Downloading Sawyer Fast Llama for ${PLATFORM}..."

        # Create bin directory
        mkdir -p "$BIN_DIR"

        # Determine download URL
        # For Windows, append .exe
        DOWNLOAD_NAME="$BINARY_NAME"
        if [[ "$PLATFORM" == windows-* ]]; then
            DOWNLOAD_NAME="${BINARY_NAME}.exe"
        fi

        URL="https://github.com/${FAST_LLAMA_REPO}/releases/download/${FAST_LLAMA_TAG}/${DOWNLOAD_NAME}"

        # Download with curl or wget
        if command -v curl &>/dev/null; then
            if curl -fsSL "$URL" -o "$DEST"; then
                ok "Downloaded ${BINARY_NAME}"
            else
                err "Download failed. Try manually:"
                echo "  ${URL}"
                DEST=""
            fi
        elif command -v wget &>/dev/null; then
            if wget -q "$URL" -O "$DEST"; then
                ok "Downloaded ${BINARY_NAME}"
            else
                err "Download failed. Try manually:"
                echo "  ${URL}"
                DEST=""
            fi
        else
            err "Neither curl nor wget found. Install one, then run:"
            echo "  mkdir -p ${BIN_DIR}"
            echo "  curl -fsSL ${URL} -o ${DEST}"
            DEST=""
        fi

        # Make executable
        if [ -n "$DEST" ] && [ -f "$DEST" ]; then
            chmod +x "$DEST"

            # Create llama-bench symlink for backward compat
            BENCH_LINK="${BIN_DIR}/llama-bench"
            if [ ! -e "$BENCH_LINK" ]; then
                ln -sf "$DEST" "$BENCH_LINK" 2>/dev/null || \
                    cp "$DEST" "$BENCH_LINK" 2>/dev/null || true
            fi
            ok "Sawyer Fast Llama ready at ${DEST}"
        fi
    fi
fi

# ── Done ──
echo ""
ok "Sawyer installed successfully!"
echo ""
echo -e "  ${BOLD}Quick start:${NC}"
echo -e "  ${CYAN}sawyer register --name my-node --gpu${NC}"
echo -e "  ${CYAN}sawyer serve${NC}"
echo -e "  ${CYAN}sawyer chat${NC}"
echo -e "  ${CYAN}sawyer bench -m /path/to/model.gguf${NC}"
echo ""