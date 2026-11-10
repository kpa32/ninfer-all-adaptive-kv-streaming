#!/usr/bin/env bash
# Configure on first run, then incrementally build the release products (Ninja, Release).
#
#   ./build.sh                 # sm_89 (the 16 GB card), 16 build threads
#   ./build.sh 86              # other architecture; "75;86;89" for several
#
# The FFmpeg and dav1d source archives must already sit in third_party/media-sources
# (cmake/MediaSources.cmake pins their URLs and digests); the build never downloads them.
set -euo pipefail
cd "$(dirname "$0")"
ARCHES=${1:-${CMAKE_CUDA_ARCHITECTURES:-89}}
BUILD_DIR=${NINFER_BUILD_DIR:-build-sm89}
JOBS=${NINFER_BUILD_JOBS:-16}
if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
  cmake -S . -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="$ARCHES" \
    -DNINFER_BUILD_APPS=ON
fi
cmake --build "$BUILD_DIR" -j "$JOBS"
