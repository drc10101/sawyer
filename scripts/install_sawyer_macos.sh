#!/usr/bin/env bash
# Sawyer — macOS Install Script
# The load is split. Friends help.
#
# ONE-LINE INSTALL:
#   curl -fsSL https://sawyer.infill.systems/install-macos.sh | bash
#
# This script:
#   1. Checks for Homebrew (installs if missing)
#   2. Checks for Python 3.11+ (installs via brew if missing)
#   3. Installs sawyer-core from PyPI (in a venv)
#   4. Downloads the Sawyer Fast Llama binary for macOS
#   5. Validates the installation
#   6. Prints next steps
#
# For Linux, use install_sawyer.sh instead.
# For Windows, use install_sawyer.ps1 instead.

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[0;33m'
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
warn()  { echo -e "  ${YELLOW}${1}${NC}"; }
err()   { echo -e "  ${RED}${1}${NC}"; }

# ── Banner ──
echo ""
echo -e "  ${BOLD}${CYAN}Sawyer${NC} ${BOLD}— Distributed MoE Inference Network (macOS)${NC}"
echo -e "  ${NC}The load is split. Friends help.${NC}"
echo ""

# ── Detect architecture ──
ARCH="$(uname -m 2>/dev/null || echo unknown)"
case "$ARCH" in
    x86_64)  PLATFORM_TAG="macos-x64" ;;
    arm64)   PLATFORM_TAG="macos-arm64" ;;
    *)
        err "Unsupported architecture: $ARCH"
        err "Sawyer requires Intel (x86_64) or Apple Silicon (arm64) macOS."
        exit 1
        ;;
esac

ok "Detected: macOS $ARCH ($PLATFORM_TAG)"

# ── Step 1: Homebrew ──
info "Step 1/4: Checking prerequisites..."

if ! command -v brew &>/dev/null; then
    warn "Homebrew not found. Installing..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

    # Add brew to PATH for this session
    if [ -f /opt/homebrew/bin/brew ]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    elif [ -f /usr/local/bin/brew ]; then
        eval "$(/usr/local/bin/brew shellenv)"
    fi

    if ! command -v brew &>/dev/null; then
        err "Homebrew installation failed."
        err "Install Homebrew manually: https://brew.sh"
        exit 1
    fi
fi
ok "Homebrew available"

# ── Step 2: Python ──
info "Step 2/4: Checking Python 3.11+..."

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
    warn "Python 3.11+ not found. Installing via Homebrew..."
    brew install python@3.12

    # Find the newly installed python
    if [ -f /opt/homebrew/bin/python3.12 ]; then
        PYTHON="/opt/homebrew/bin/python3.12"
    elif [ -f /usr/local/bin/python3.12 ]; then
        PYTHON="/usr/local/bin/python3.12"
    else
        PYTHON="python3"
    fi

    if ! command -v "$PYTHON" &>/dev/null; then
        err "Python installation failed."
        err "Install Python manually: brew install python@3.12"
        exit 1
    fi
fi

PYVER=$($PYTHON --version 2>&1)
ok "Using $PYVER"

# ── Step 3: sawyer-core ──
info "Step 3/4: Installing sawyer-core..."

# Create venv for isolation
VENV_DIR="${HOME}/.sawyer/venv"
if [ ! -f "${VENV_DIR}/bin/activate" ]; then
    info "Creating virtual environment..."
    $PYTHON -m venv "$VENV_DIR"
fi

# Activate venv
source "${VENV_DIR}/bin/activate"
PYTHON="python"
PIP="pip"

# Upgrade pip
$PYTHON -m pip install --upgrade pip --quiet

# Install sawyer-core
$PIP install --upgrade "$PKG" 2>&1 || {
    err "pip install failed. Try:"
    echo "  $PYTHON -m pip install --user sawyer-core"
    exit 1
}
ok "sawyer-core installed"

# Create wrapper script
mkdir -p "${BIN_DIR}"
cat > "${BIN_DIR}/sawyer-agent" << WRAPPER
#!/usr/bin/env bash
source "${VENV_DIR}/bin/activate"
exec python -m sawyer.cli "\$@"
WRAPPER
chmod +x "${BIN_DIR}/sawyer-agent"

# ── Step 4: Sawyer Fast Llama binary ──
info "Step 4/4: Downloading Sawyer Fast Llama..."

# macOS: no pre-built binaries yet
BINARY_NAME="sawyer-fast-llama-${PLATFORM_TAG}"
DEST="${BIN_DIR}/${BINARY_NAME}"

info "macOS Fast Llama binary not yet available for download."
info "sawyer bench will use llama.cpp from Homebrew if installed."
warn "Install llama.cpp: brew install llama.cpp"

ok "Fast Llama: will use system llama.cpp"

# ── Add to PATH ──
SHELL_RC="${HOME}/.zshrc"
case "$SHELL" in
    *bash*) SHELL_RC="${HOME}/.bashrc" ;;
esac

PATH_LINE="export PATH=\"${BIN_DIR}:\$PATH\""
if ! grep -q "${BIN_DIR}" "$SHELL_RC" 2>/dev/null; then
    echo "" >> "$SHELL_RC"
    echo "# Sawyer" >> "$SHELL_RC"
    echo "$PATH_LINE" >> "$SHELL_RC"
    info "Added ${BIN_DIR} to PATH in ${SHELL_RC}"
    info "Run 'source ${SHELL_RC}' or open a new terminal."
fi

# ── Validate ──
info "Validating installation..."

VALIDATION_ERRORS=0

SAWYER_BIN="${BIN_DIR}/sawyer-agent"
if [ -f "$SAWYER_BIN" ]; then
    VERSION_CHECK=$("$SAWYER_BIN" --help 2>&1 | head -1 || true)
    if echo "$VERSION_CHECK" | grep -qi "sawyer"; then
        ok "sawyer-agent command works"
    else
        warn "sawyer-agent --help returned unexpected output"
        VALIDATION_ERRORS=$((VALIDATION_ERRORS + 1))
    fi
else
    if $PYTHON -m sawyer.cli --help &>/dev/null; then
        ok "sawyer-agent works via python -m sawyer"
    else
        err "sawyer-agent command not found"
        VALIDATION_ERRORS=$((VALIDATION_ERRORS + 1))
    fi
fi

# Check Python imports
$PYTHON -c "from sawyer.config import SawyerConfig; c = SawyerConfig(); print(f'  Config OK: router={c.router_url}')" 2>/dev/null && \
    ok "Python imports work" || {
    err "Python import validation failed"
    VALIDATION_ERRORS=$((VALIDATION_ERRORS + 1))
}

# ── Results ──
echo ""
if [ "$VALIDATION_ERRORS" -eq 0 ]; then
    ok "Sawyer installed successfully!"
else
    warn "Sawyer installed with $VALIDATION_ERRORS warning(s). See above."
fi

echo ""
echo -e "  ${BOLD}Quick start:${NC}"
echo -e "  ${CYAN}sawyer-agent chat${NC}      Start the chat client (web UI at http://localhost:8000)"
echo -e "  ${CYAN}sawyer-agent serve${NC}     Start serving expert inference requests"
echo -e "  ${CYAN}sawyer-agent run${NC}       One command: start Sawyer + Ollama + agent"
echo ""
echo -e "  ${YELLOW}macOS: For GPU acceleration, install Ollama from https://ollama.com${NC}"
echo -e "  ${YELLOW}       Then: ollama pull llama3 && sawyer-agent run${NC}"
echo ""

if [ "$VALIDATION_ERRORS" -gt 0 ]; then
    echo -e "  ${YELLOW}If 'sawyer-agent' command not found, run:${NC}"
    echo -e "  ${CYAN}source ${SHELL_RC}${NC}"
    echo ""
fi