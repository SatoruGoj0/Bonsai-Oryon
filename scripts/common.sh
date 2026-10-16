# Shared settings for Bonsai-Oryon scripts (sourced, not executed).
BONSAI_VERSION="0.3.0"
UPSTREAM_URL="https://github.com/PrismML-Eng/llama.cpp.git"
UPSTREAM_BRANCH="prism"
UPSTREAM_COMMIT="e3de715222b2667c54a267441159a5729f11155e"
BONSAI_LIBDIR="/usr/lib/bonsai-oryon"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUBMODULE_DIR="${REPO_ROOT}/upstream/llama.cpp"
BUILD_ROOT="${BUILD_ROOT:-${REPO_ROOT}/build}"
SRC_DIR="${BUILD_ROOT}/src"

log() { printf '\033[1;34m[bonsai]\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31m[bonsai] error:\033[0m %s\n' "$*" >&2; exit 1; }
