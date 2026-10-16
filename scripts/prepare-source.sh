#!/usr/bin/env bash
# Export the pinned PrismML llama.cpp commit into build/src and apply patches/*.patch.
# The submodule itself is never modified, so the source tree stays auditable.
#   PATCH_SET=all       (default) every patch
#   PATCH_SET=baseline  only the build/loader patches (0003, 0004): upstream kernels, identical CPU-variant
#                       selection and library layout -> isolates the effect of the kernel patches in A/B runs
set -euo pipefail
source "$(dirname "$0")/common.sh"

[ -e "${SUBMODULE_DIR}/.git" ] || die "submodule missing: run 'git submodule update --init'"
have="$(git -C "${SUBMODULE_DIR}" rev-parse HEAD)"
[ "${have}" = "${UPSTREAM_COMMIT}" ] || die "submodule is at ${have}, expected pinned ${UPSTREAM_COMMIT}"

rm -rf "${SRC_DIR}"
mkdir -p "${SRC_DIR}"
log "exporting ${UPSTREAM_COMMIT} -> ${SRC_DIR}"
git -C "${SUBMODULE_DIR}" archive --format=tar "${UPSTREAM_COMMIT}" | tar -x -C "${SRC_DIR}"

PATCH_SET="${PATCH_SET:-all}"
for p in "${REPO_ROOT}"/patches/*.patch; do
    if [ "${PATCH_SET}" = baseline ]; then
        case "$(basename "$p")" in 0003-*|0004-*) ;; *) log "skipping $(basename "$p") (baseline)"; continue;; esac
    fi
    log "applying $(basename "$p")"
    (cd "${SRC_DIR}" && git apply --check "$p" && git apply --whitespace=nowarn "$p") || die "patch failed: $p"
done

# record provenance for the package
{
    echo "upstream: ${UPSTREAM_URL} (${UPSTREAM_BRANCH})"
    echo "commit:   ${UPSTREAM_COMMIT}"
    echo "patch set: ${PATCH_SET}"
    echo "patches:"
    (cd "${REPO_ROOT}" && sha256sum patches/*.patch | sed 's/^/  /')
} > "${SRC_DIR}/BONSAI_PROVENANCE.txt"
log "source ready"
