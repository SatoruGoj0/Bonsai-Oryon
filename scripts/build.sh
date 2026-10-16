#!/usr/bin/env bash
# Configure and build the CPU-only, runtime-dispatched ARM64 engine from build/src.
#   scripts/build.sh                 # release build -> build/release
#   BONSAI_VARIANT=baseline ...      # (used by CI) build the UNPATCHED pinned upstream for comparison
# Extra CMake arguments can be passed via CMAKE_EXTRA.
set -euo pipefail
source "$(dirname "$0")/common.sh"

OUT="${OUT:-${BUILD_ROOT}/release}"
JOBS="${JOBS:-$(nproc)}"
[ -f "${SRC_DIR}/CMakeLists.txt" ] || die "run scripts/prepare-source.sh first"

GEN=()
command -v ninja >/dev/null && GEN=(-G Ninja)

# CPU-only: every GPU backend is off (Vulkan in particular is excluded by design).
# Threading: since v0.3 the default CPU modules use ggml's persistent threadpool (GGML_OPENMP=OFF); the OpenMP
# modules of v0.1/v0.2 are built as well (BUILD_OMP_VARIANT=1) into bin/omp/ for like-for-like A/B tests.
# GGML_CPU_ALL_VARIANTS builds one libggml-cpu-<variant>.so per ARM feature level; at start-up the
# loader scores each against HWCAP/HWCAP2 and loads the best one the CPU can run. Nothing is compiled
# for "this" machine (GGML_NATIVE=OFF), so the binaries run on any ARMv8.0+ CPU.
cmake -S "${SRC_DIR}" -B "${OUT}" "${GEN[@]}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr \
    -DBUILD_SHARED_LIBS=ON \
    -DGGML_NATIVE=OFF \
    -DGGML_BACKEND_DL=ON \
    -DGGML_CPU_ALL_VARIANTS=ON \
    -DGGML_BACKEND_DIR="${BONSAI_LIBDIR}" \
    -DGGML_OPENMP="${BONSAI_OPENMP:-OFF}" \
    -DGGML_LLAMAFILE=ON \
    -DGGML_VULKAN=OFF -DGGML_CUDA=OFF -DGGML_METAL=OFF -DGGML_OPENCL=OFF -DGGML_SYCL=OFF \
    -DGGML_HIP=OFF -DGGML_RPC=OFF -DGGML_BLAS=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_APP=OFF \
    -DLLAMA_BUILD_UI=OFF \
    -DLLAMA_USE_PREBUILT_UI=OFF \
    -DLLAMA_OPENSSL=OFF \
    -DLLAMA_BUILD_NUMBER=0 \
    -DLLAMA_BUILD_COMMIT="${UPSTREAM_COMMIT:0:9}+bonsai-oryon-${BONSAI_VERSION}" \
    -DCMAKE_INSTALL_RPATH="${BONSAI_LIBDIR}" \
    -DCMAKE_SHARED_LINKER_FLAGS="-Wl,--disable-new-dtags -Wl,-z,relro -Wl,-z,now" \
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,--disable-new-dtags -Wl,-z,relro -Wl,-z,now" \
    ${CMAKE_EXTRA:-}

# the ggml-cpu-<variant> modules are dependencies of ggml, so they are always built
BUILD_TARGETS="${BUILD_TARGETS:-llama-cli llama-server llama-bench llama-perplexity llama-quantize}"
cmake --build "${OUT}" -j "${JOBS}" --target ${BUILD_TARGETS}

log "built into ${OUT}/bin"
ls -1 "${OUT}/bin" | sed 's/^/  /' >&2

# OpenMP variant of the CPU backend modules (same sources and flags, GGML_OPENMP=ON), loaded with
# BONSAI_BACKEND_DIR=<libdir>/omp
if [ "${BUILD_OMP_VARIANT:-1}" = 1 ] && [ "${BONSAI_OPENMP:-OFF}" = OFF ]; then
    log "building the OpenMP variant of the CPU backend modules"
    OUT="${OUT}-omp" BONSAI_OPENMP=ON BUILD_OMP_VARIANT=0 BUILD_DIAG=0 BUILD_TARGETS=ggml "$0"
    mkdir -p "${OUT}/bin/omp"
    cp "${OUT}-omp"/bin/libggml-cpu-*.so "${OUT}/bin/omp/"
    log "OpenMP CPU modules in ${OUT}/bin/omp"
fi

# bonsai-diag (memory bandwidth + PQ2_0 GEMV diagnostics), linked against this build's shared ggml
if [ "${BUILD_DIAG:-1}" = 1 ]; then
    cmake -S "${REPO_ROOT}/tools/diag" -B "${OUT}-diag" "${GEN[@]}" -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_SRC="${SRC_DIR}" -DLLAMA_BUILD_DIR="${OUT}" ${CMAKE_EXTRA:-}
    cmake --build "${OUT}-diag" -j "${JOBS}"
    cp "${OUT}-diag/bonsai-diag" "${OUT}/bin/bonsai-diag"
    log "built ${OUT}/bin/bonsai-diag"
fi
