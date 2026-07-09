#!/usr/bin/env bash
# Sawyer Fast Llama — RunPod build script
# Paste this entire file into the RunPod web terminal (Connect → HTTP 8080 → Terminal)
# Wait for "ALL DONE" at the bottom. Then download the two files from /root/

set -e

echo "============================================"
echo " Sawyer Fast Llama — Build Script"
echo "============================================"
echo ""

# 1. Install build dependencies
echo "[1/5] Installing build tools..."
apt update -qq && apt install -y -qq git cmake gcc g++ curl > /dev/null 2>&1

# 2. Clone the Sawyer-optimized llama.cpp fork
echo "[2/5] Cloning llama.cpp (sawyer/optimize-moe-prefetch branch)..."
if [ ! -d /root/llama.cpp ]; then
    git clone https://github.com/drc10101/llama.cpp.git /root/llama.cpp
fi
cd /root/llama.cpp
git checkout sawyer/optimize-moe-prefetch

# 3. Build with CUDA
echo "[3/5] Building with CUDA support (this takes 5-10 minutes)..."
mkdir -p build && cd build
cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -5
cmake --build . --config Release -j$(nproc) 2>&1 | tail -5

# 4. Verify binaries
echo "[4/5] Verifying binaries..."
ls -la bin/llama-bench bin/llama-server
echo "  llama-bench:  $(du -h bin/llama-bench | cut -f1)"
echo "  llama-server:  $(du -h bin/llama-server | cut -f1)"

# 5. Copy to /root/ for easy download
echo "[5/5] Copying to /root/ for download..."
cp bin/llama-bench /root/sawyer-fast-llama-linux-x64
cp bin/llama-server /root/sawyer-fast-llama-cli-linux-x64
chmod +x /root/sawyer-fast-llama-*

echo ""
echo "============================================"
echo " ALL DONE"
echo "============================================"
echo ""
echo " Two files ready in /root/:"
echo "   sawyer-fast-llama-linux-x64"
echo "   sawyer-fast-llama-cli-linux-x64"
echo ""
echo " Download them from the Jupyter file browser:"
echo "   1. Look at the left sidebar in Jupyter"
echo "   2. Navigate to /root/"
echo "   3. Right-click each file -> Download"
echo ""
echo " Then stop this pod in the RunPod dashboard."
echo "============================================"