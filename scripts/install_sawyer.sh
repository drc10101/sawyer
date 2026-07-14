#!/usr/bin/env bash
# Sawyer — Distributed MoE Inference Network
# The load is split. Friends help.
#
# ONE-LINE INSTALL:
#   Linux:   curl -fsSL https://sawyer.infill.systems/install.sh | bash
#   macOS:   curl -fsSL https://sawyer.infill.systems/install.sh | bash
#
# Or run locally:
#   ./install_sawyer.sh
#
# This script:
#   1. Checks for Python 3.11+ (installs it if missing)
#   2. Installs sawyer-core from PyPI
#   3. Downloads the Sawyer Fast Llama binary for your platform
#   4. Validates the installation
#   5. Prints next steps
#
# Supports: Ubuntu/Debian, Fedora/RHEL/CentOS, macOS
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
echo -e "  ${BOLD}${CYAN}Sawyer${NC} ${BOLD}— Distributed MoE Inference Network${NC}"
echo -e "  ${NC}The load is split. Friends help.${NC}"
echo ""

# ── Detect OS ──
OS="$(uname -s 2>/dev/null || echo unknown)"
ARCH="$(uname -m 2>/dev/null || echo unknown)"

case "$OS" in
    Linux)  PLATFORM="linux" ;;
    Darwin) PLATFORM="macos" ;;
    *)      err "Unsupported OS: $OS"; err "Use Windows install_sawyer.ps1 instead."; exit 1 ;;
esac

# ── Step 1: Python ──
info "Step 1/4: Checking Python 3.11+..."

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
    warn "Python 3.11+ not found. Installing..."

    if [ "$PLATFORM" = "linux" ]; then
        # Detect distro
        if [ -f /etc/debian_version ] || command -v apt-get &>/dev/null; then
            # Ubuntu/Debian
            info "Installing Python 3.12 via apt..."
            sudo apt-get update -qq
            sudo apt-get install -y -qq python3.12 python3.12-venv python3-pip 2>/dev/null || {
                # Fallback: add deadsnakes PPA for older Ubuntu
                sudo apt-get install -y -qq software-properties-common 2>/dev/null || true
                sudo add-apt-repository -y ppa:deadsnakes/ppa 2>/dev/null || true
                sudo apt-get update -qq 2>/dev/null || true
                sudo apt-get install -y -qq python3.12 python3.12-venv python3-pip
            }
            PYTHON="python3.12"
        elif [ -f /etc/fedora-release ] || command -v dnf &>/dev/null; then
            # Fedora/RHEL/CentOS
            info "Installing Python 3.12 via dnf..."
            sudo dnf install -y python3.12 python3.12-pip 2>/dev/null || \
                sudo dnf install -y python3 python3-pip
            PYTHON="python3.12"
        elif command -v yum &>/dev/null; then
            # Older RHEL/CentOS
            info "Installing Python 3 via yum..."
            sudo yum install -y python3 python3-pip
            PYTHON="python3"
        else
            err "Cannot determine Linux distro. Install Python 3.11+ manually:"
            echo "  Ubuntu/Debian: sudo apt install python3.12 python3.12-venv"
            echo "  Fedora: sudo dnf install python3.12"
            echo "  Then re-run this script."
            exit 1
        fi
    elif [ "$PLATFORM" = "macos" ]; then
        # macOS: use Homebrew
        if ! command -v brew &>/dev/null; then
            info "Installing Homebrew..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        info "Installing Python via Homebrew..."
        brew install python@3.12
        PYTHON="python3.12"
    fi

    # Verify Python installed
    if ! command -v "$PYTHON" &>/dev/null; then
        err "Python installation failed."
        err "Install Python 3.11+ manually and re-run this script."
        exit 1
    fi
fi

PYVER=$($PYTHON --version 2>&1)
ok "Using $PYVER"

# ── Step 2: pip + venv ──
info "Step 2/4: Installing sawyer-core..."

# Ensure pip is available
$PYTHON -m pip --version &>/dev/null || {
    info "Installing pip..."
    $PYTHON -m ensurepip --upgrade 2>/dev/null || {
        # Bootstrap pip
        curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
        $PYTHON /tmp/get-pip.py --user
        rm -f /tmp/get-pip.py
    }
}

# Create a venv in ~/.sawyer for isolation
VENV_DIR="${HOME}/.sawyer/venv"
if [ ! -f "${VENV_DIR}/bin/activate" ]; then
    info "Creating virtual environment..."
    $PYTHON -m venv "$VENV_DIR" 2>/dev/null || {
        # If venv module is missing, install to user site
        warn "venv unavailable, installing to user site-packages"
        $PYTHON -m pip install --user --upgrade "$PKG"
        # Set PYTHON to the one with the package
        PYTHON="$PYTHON"
        # Create a wrapper script instead
        mkdir -p "${BIN_DIR}"
        cat > "${BIN_DIR}/sawyer-agent" << 'WRAPPER'
#!/usr/bin/env bash
exec python3 -m sawyer.cli "$@"
WRAPPER
        chmod +x "${BIN_DIR}/sawyer-agent"
        SAWYER_CMD="${BIN_DIR}/sawyer-agent"
    }
fi

if [ -f "${VENV_DIR}/bin/activate" ]; then
    # Activate venv for the rest of the script
    source "${VENV_DIR}/bin/activate"
    PYTHON="python"
    PIP="pip"

    # Upgrade pip in venv
    $PYTHON -m pip install --upgrade pip --quiet

    # Install sawyer-core
    $PIP install --upgrade "$PKG" 2>&1 || {
        err "pip install failed. Try:"
        echo "  $PYTHON -m pip install --user sawyer-core"
        exit 1
    }

    # Create wrapper script pointing to venv
    mkdir -p "${BIN_DIR}"
    cat > "${BIN_DIR}/sawyer-agent" << WRAPPER
#!/usr/bin/env bash
source "${VENV_DIR}/bin/activate"
exec python -m sawyer.cli "\$@"
WRAPPER
    chmod +x "${BIN_DIR}/sawyer-agent"
    SAWYER_CMD="${BIN_DIR}/sawyer-agent"
fi

ok "sawyer-core installed"

# ── Step 3: Sawyer Fast Llama binary ──
info "Step 3/4: Downloading Sawyer Fast Llama..."

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

PLATFORM_TAG=$(detect_platform)

if [ "$PLATFORM_TAG" = "unsupported" ]; then
    warn "Sawyer Fast Llama: skipping binary (unsupported platform: $OS $ARCH)"
    warn "sawyer bench will use system llama-bench if available"
else
    BINARY_NAME="sawyer-fast-llama-${PLATFORM_TAG}"
    DEST="${BIN_DIR}/${BINARY_NAME}"

    # Download platform-appropriate binary
    # Available release assets:
    #   Linux x64 CPU:  sawyer-fast-llama-linux-x64 (on llama.cpp repo)
    #   Linux x64 CUDA: sawyer-fast-llama-linux-x64-cuda.tar.gz (on sawyer repo)
    #   Linux x64 CLI:  sawyer-fast-llama-cli-linux-x64 (on llama.cpp repo)
    #   Windows x64:    sawyer-fast-llama-cli-windows-x64.zip (on llama.cpp repo)
    #   Windows CUDA:   sawyer-fast-llama-windows-cuda.zip (on llama.cpp repo)
    #   macOS:          Not yet available — sawyer bench will use system llama

    if [ "$PLATFORM_TAG" = "macos-x64" ] || [ "$PLATFORM_TAG" = "macos-arm64" ]; then
        # No macOS binary in the release yet
        info "macOS Fast Llama binary not yet available for download."
        info "sawyer bench will use llama.cpp from Homebrew if installed."
        warn "Install llama.cpp: brew install llama.cpp"
        BINARY_NAME=""
        DEST=""
    fi

    # Try CUDA variant first (Linux x64 only)
    if [ "$PLATFORM_TAG" = "linux-x64" ]; then
        CUDA_TAR="sawyer-fast-llama-linux-x64-cuda.tar.gz"
        CUDA_URL="https://github.com/${REPO}/releases/download/v${VERSION}/${CUDA_TAR}"
        CUDA_DEST="${BIN_DIR}/sawyer-fast-llama-linux-x64-cuda"

        # Check for NVIDIA GPU
        HAS_NVIDIA=false
        if command -v nvidia-smi &>/dev/null; then
            nvidia-smi &>/dev/null && HAS_NVIDIA=true
        fi

        if [ "$HAS_NVIDIA" = true ]; then
            if [ ! -f "${CUDA_DEST}/llama-server" ]; then
                info "NVIDIA GPU detected — downloading CUDA binary..."
                mkdir -p "${BIN_DIR}"
                if command -v curl &>/dev/null; then
                    if curl -fsSL "$CUDA_URL" -o /tmp/sawyer-cuda.tar.gz; then
                        mkdir -p "${CUDA_DEST}"
                        tar xzf /tmp/sawyer-cuda.tar.gz -C "${CUDA_DEST}" 2>/dev/null || true
                        rm -f /tmp/sawyer-cuda.tar.gz
                        ok "CUDA binary installed"
                    else
                        warn "CUDA download failed. CPU-only binary will be used."
                    fi
                fi
            else
                ok "CUDA binary already cached"
            fi
        fi
    fi

    # CPU binary
    if [ -f "$DEST" ]; then
        ok "Fast Llama already cached at ${DEST}"
    else
        mkdir -p "$BIN_DIR"

        # Linux CPU binary is on the llama.cpp repo
        if [ "$PLATFORM_TAG" = "linux-x64" ]; then
            DOWNLOAD_NAME="sawyer-fast-llama-linux-x64"
            URL="https://github.com/${FAST_LLAMA_REPO}/releases/download/${FAST_LLAMA_TAG}/${DOWNLOAD_NAME}"
        else
            DOWNLOAD_NAME="$BINARY_NAME"
            URL="https://github.com/${FAST_LLAMA_REPO}/releases/download/${FAST_LLAMA_TAG}/${DOWNLOAD_NAME}"
        fi

        if command -v curl &>/dev/null; then
            if curl -fsSL "$URL" -o "$DEST"; then
                chmod +x "$DEST"
                ok "Downloaded ${BINARY_NAME}"
            else
                err "Download failed. Try manually:"
                echo "  ${URL}"
                DEST=""
            fi
        elif command -v wget &>/dev/null; then
            if wget -q "$URL" -O "$DEST"; then
                chmod +x "$DEST"
                ok "Downloaded ${BINARY_NAME}"
            else
                err "Download failed. Try manually:"
                echo "  ${URL}"
                DEST=""
            fi
        else
            err "Neither curl nor wget found."
            echo "  Install one, then run: mkdir -p ${BIN_DIR} && curl -fsSL ${URL} -o ${DEST}"
            DEST=""
        fi

        # Create llama-bench symlink
        if [ -n "$DEST" ] && [ -f "$DEST" ]; then
            BENCH_LINK="${BIN_DIR}/llama-bench"
            ln -sf "$DEST" "$BENCH_LINK" 2>/dev/null || \
                cp "$DEST" "$BENCH_LINK" 2>/dev/null || true
        fi
    fi
fi

ok "Fast Llama ready"

# ── Step 4: Validate ──
info "Step 4/4: Validating installation..."

# Add bin to PATH in shell config
SHELL_RC="${HOME}/.bashrc"
case "$SHELL" in
    *zsh*) SHELL_RC="${HOME}/.zshrc" ;;
esac

PATH_LINE="export PATH=\"${BIN_DIR}:\$PATH\""
if ! grep -q "${BIN_DIR}" "$SHELL_RC" 2>/dev/null; then
    echo "" >> "$SHELL_RC"
    echo "# Sawyer" >> "$SHELL_RC"
    echo "$PATH_LINE" >> "$SHELL_RC"
    info "Added ${BIN_DIR} to PATH in ${SHELL_RC}"
    info "Run 'source ${SHELL_RC}' or open a new terminal to update PATH."
fi

# Validate sawyer command
VALIDATION_ERRORS=0

if [ -f "${VENV_DIR}/bin/activate" ]; then
    source "${VENV_DIR}/bin/activate"
fi
SAWYER_BIN="${BIN_DIR}/sawyer-agent"
if [ -f "$SAWYER_BIN" ]; then
    VERSION_CHECK=$("$SAWYER_BIN" --help 2>&1 | head -1 || true)
    if echo "$VERSION_CHECK" | grep -qi "sawyer"; then
        ok "sawyer-agent command works"
    else
        warn "sawyer-agent --help returned unexpected output"
        warn "  $VERSION_CHECK"
        VALIDATION_ERRORS=$((VALIDATION_ERRORS + 1))
    fi
else
    # Try python -m sawyer as fallback
    if $PYTHON -m sawyer.cli --help &>/dev/null; then
        ok "sawyer-agent works via python -m sawyer"
    else
        err "sawyer-agent command not found and python -m sawyer failed"
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
echo -e "  ${CYAN}sawyer-agent bench${NC}     Benchmark MoE prefill speedup"
echo ""

if [ "$PLATFORM" = "macos" ]; then
    echo -e "  ${YELLOW}macOS: For GPU acceleration, install Ollama from https://ollama.com${NC}"
    echo -e "  ${YELLOW}       Then: ollama pull llama3 && sawyer-agent run${NC}"
    echo ""
fi

if [ "$VALIDATION_ERRORS" -gt 0 ]; then
    echo -e "  ${YELLOW}If 'sawyer-agent' command not found, run:${NC}"
    echo -e "  ${CYAN}source ${SHELL_RC}${NC}"
    echo ""
fi