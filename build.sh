#!/usr/bin/env bash
# =============================================================================
# NInfer build script — RTX 4080 SUPER (sm_89) / CUDA 13.4 / Debian 13
# Build only: no git, no model downloads, no source edits.
#   ./build.sh          incremental build (recommended)
#   ./build.sh --clean  delete the build dir, reconfigure + full build
# Adapted from the verified ~/ninfer-fork/build.sh of the kv-stream-27b-16g line.
# =============================================================================
set -euo pipefail
# --- Proxy (the build itself is offline; NInfer downloads no dependencies.
#---   Only useful for manual `git pull`; direct GitHub access works on this LAN.)
# export HTTP_PROXY="http://192.168.2.12:7890"
# export HTTPS_PROXY="http://192.168.2.12:7890"
# export NO_PROXY="localhost,127.0.0.1,::1"
# --- CUDA paths ---
export CUDA_HOME="/usr/local/cuda"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CUDACXX="$CUDA_HOME/bin/nvcc"
# --- Configuration ---
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$REPO_DIR/build-sm89"
CUDA_ARCH="89"          # RTX 4080 SUPER = sm_89, do not change to 120a
JOBS="${JOBS:-16}"      # 16 cores but tight RAM; use 4 while a model server is running
clean=0
[[ "${1:-}" == "--clean" ]] && clean=1
echo "============================================"
echo "NInfer build: sm_$CUDA_ARCH / $CUDA_HOME"
echo "Source: $REPO_DIR"
echo "Output: $BUILD_DIR"
echo "============================================"
# [1/5] Build tools
echo "[1/5] Checking build tools..."
missing=()
for t in cmake ninja git pkg-config; do
    command -v "$t" >/dev/null 2>&1 || missing+=("$t")
done
if ((${#missing[@]})); then
    echo "Missing tools: ${missing[*]}"
    echo "  sudo apt update && sudo apt install -y cmake ninja-build git pkg-config ccache"
    exit 1
fi
echo "  cmake $(cmake --version | head -1)"
echo "  ninja $(ninja --version)"
# [2/5] CUDA and host compiler
echo "[2/5] Checking CUDA and compiler..."
[[ -x "$CUDACXX" ]] || { echo "Cannot find $CUDACXX"; exit 1; }
nvcc_ver="$("$CUDACXX" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')"
echo "  nvcc $nvcc_ver"
# CUDA 12.8 is the hard floor (CMakeLists.txt fatals below it)
awk -v v="$nvcc_ver" 'BEGIN{split(v,a,"."); exit !(a[1]*100+a[2] >= 1208)}' \
    || { echo "CUDA $nvcc_ver < 12.8, unsupported"; exit 1; }
# gcc-13 is the host compiler the docs qualified; fall back to the system gcc
# (CUDA 13.x also supports gcc-14).
if command -v gcc-13 >/dev/null 2>&1 && command -v g++-13 >/dev/null 2>&1; then
    CC_BIN="gcc-13"; CXX_BIN="g++-13"
else
    CC_BIN="gcc"; CXX_BIN="g++"
fi
echo "  host compiler: $CXX_BIN ($($CXX_BIN --version | head -1))"
# [3/5] Media sources (the build never downloads; the archives are pinned in
# cmake/MediaSources.cmake and must already sit in third_party/media-sources)
for a in ffmpeg-9.0.2.tar.xz dav1d-1.5.4.tar.xz; do
    [[ -f "$REPO_DIR/third_party/media-sources/$a" ]] || {
        echo "Missing third_party/media-sources/$a; copy it from a sibling tree."
        exit 1
    }
done
# [4a] Built-in WebUI: prebuilt llama.cpp pages compiled into ninfer-serve.
# third_party/webui is gitignored; take it from a sibling tree when absent.
WEBUI_DIR=""
for d in "$REPO_DIR"/third_party/webui/*/*/ "$REPO_DIR"/third_party/webui/*/; do
    if [[ -f "$d/index.html" || -f "$d/index.html.gz" ]]; then WEBUI_DIR="${d%/}"; break; fi
done
if [[ -z "$WEBUI_DIR" ]]; then
    for s in /home/kpa/ninfer-fork/ninfer-all-adaptive-kv-streaming /home/kpa/ninfer/ninfer-all; do
        if cp -rn "$s/third_party/webui" "$REPO_DIR/third_party/" 2>/dev/null; then
            for d in "$REPO_DIR"/third_party/webui/*/*/ "$REPO_DIR"/third_party/webui/*/; do
                if [[ -f "$d/index.html" || -f "$d/index.html.gz" ]]; then WEBUI_DIR="${d%/}"; break; fi
            done
            [[ -n "$WEBUI_DIR" ]] && break
        fi
    done
fi
echo "  WebUI: ${WEBUI_DIR:-<none>}"
# [4/5] Configure
if ((clean)); then
    echo "[4/5] Cleaning $BUILD_DIR ..."
    rm -rf -- "$BUILD_DIR"
fi
echo "[4/5] Configuring CMake..."
launcher_opts=()
command -v ccache >/dev/null 2>&1 && launcher_opts=(
    -DCMAKE_C_COMPILER_LAUNCHER=ccache
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
    -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache)
cmake -S "$REPO_DIR" -B "$BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="$CC_BIN" \
    -DCMAKE_CXX_COMPILER="$CXX_BIN" \
    -DCMAKE_CUDA_COMPILER="$CUDACXX" \
    -DCMAKE_CUDA_HOST_COMPILER="$CXX_BIN" \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DNINFER_BUILD_APPS=ON \
    -DNINFER_WEBUI_DIR="$WEBUI_DIR" \
    "${launcher_opts[@]}"
# [5/5] Build
echo "[5/5] Building (-j$JOBS)..."
cmake --build "$BUILD_DIR" -j "$JOBS"
echo "============================================"
echo "Build finished, products:"
for b in ninfer ninfer-serve ninfer-calibrate ninfer-perplexity; do
    [[ -x "$BUILD_DIR/apps/$b" ]] && echo "  $BUILD_DIR/apps/$b"
done
echo "============================================"
