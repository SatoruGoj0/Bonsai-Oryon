#!/usr/bin/env bash
# Assemble bonsai-oryon_<version>_arm64.deb from a finished build (scripts/build.sh).
#   packaging/build-deb.sh [build-dir] [out-dir]
# Layout:
#   /usr/bin/bonsai-{cli,server,bench}        real executables, DT_RPATH=/usr/lib/bonsai-oryon
#   /usr/lib/bonsai-oryon/bin/bonsai-perplexity  helper for output comparisons (not on $PATH)
#   /usr/lib/bonsai-oryon/bin/bonsai-diag        memory-bandwidth / GEMV diagnostics (not on $PATH)
#   /usr/share/bonsai-oryon/                  benchmark / comparison scripts
#   /usr/lib/bonsai-oryon/                    private libllama/libggml*/libmtmd + CPU variant modules
#   /usr/lib/bonsai-oryon/omp/                CPU variant modules built with OpenMP (v0.2 threading), selected
#                                             with BONSAI_BACKEND_DIR=/usr/lib/bonsai-oryon/omp
#   /usr/bin/bonsai-tune                      writes the per-user tuned profile (~/.config/bonsai-oryon/profile)
#   /usr/share/doc/bonsai-oryon/              README, changelog, copyright, upstream licenses, provenance
# Nothing outside these paths is touched, no maintainer scripts, no services.
set -euo pipefail
source "$(dirname "$0")/../scripts/common.sh"

BUILD="${1:-${BUILD_ROOT}/release}"
OUTDIR="${2:-${BUILD_ROOT}/dist}"
PKG=bonsai-oryon
ARCH=arm64
STAGE="${BUILD_ROOT}/pkgroot"
DEB="${OUTDIR}/${PKG}_${BONSAI_VERSION}_${ARCH}.deb"

[ -x "${BUILD}/bin/llama-cli" ] || die "no build found in ${BUILD}"
rm -rf "${STAGE}"; mkdir -p "${STAGE}/DEBIAN" "${STAGE}/usr/bin" "${STAGE}${BONSAI_LIBDIR}" "${STAGE}/usr/share/doc/${PKG}" "${OUTDIR}"

install -m 0755 "${BUILD}/bin/llama-cli"    "${STAGE}/usr/bin/bonsai-cli"
install -m 0755 "${BUILD}/bin/llama-server" "${STAGE}/usr/bin/bonsai-server"
install -m 0755 "${BUILD}/bin/llama-bench"  "${STAGE}/usr/bin/bonsai-bench"
# helper tool for numerical comparisons on the target (kept off $PATH) + benchmark scripts
install -D -m 0755 "${BUILD}/bin/llama-perplexity" "${STAGE}${BONSAI_LIBDIR}/bin/bonsai-perplexity"
install -D -m 0755 "${BUILD}/bin/bonsai-diag"      "${STAGE}${BONSAI_LIBDIR}/bin/bonsai-diag"
SHARE="${STAGE}/usr/share/${PKG}"
install -D -m 0755 "${REPO_ROOT}/bench/bonsai-benchmark.sh"        "${SHARE}/bonsai-benchmark.sh"
install -D -m 0755 "${REPO_ROOT}/bench/compare-output.sh"          "${SHARE}/compare-output.sh"
install -D -m 0755 "${REPO_ROOT}/tests/numerics/compare-kld.sh"    "${SHARE}/compare-kld.sh"
install -D -m 0755 "${REPO_ROOT}/bench/bonsai-bench-report.py"     "${SHARE}/bonsai-bench-report.py"
install -D -m 0755 "${REPO_ROOT}/bench/bonsai-release-run.sh"      "${SHARE}/bonsai-release-run.sh"
install -D -m 0755 "${REPO_ROOT}/bench/bonsai-tune.sh"             "${STAGE}/usr/bin/bonsai-tune"

# private shared libraries: copy the real files and recreate the SONAME symlinks
shopt -s nullglob
for f in "${BUILD}"/bin/lib*.so* ; do
    if [ -L "$f" ]; then cp -P "$f" "${STAGE}${BONSAI_LIBDIR}/"; else install -m 0644 "$f" "${STAGE}${BONSAI_LIBDIR}/"; fi
done
# drop development symlinks (lib*.so without version) that nothing loads at runtime, keep modules
for f in "${STAGE}${BONSAI_LIBDIR}"/lib*.so; do
    case "$(basename "$f")" in libggml-cpu-*.so) ;; *) [ -L "$f" ] && rm -f "$f" ;; esac
done
# tool implementation libraries: ship only those the packaged executables use
for f in "${STAGE}${BONSAI_LIBDIR}"/libllama-*-impl.so; do
    case "$(basename "$f")" in libllama-cli-impl.so|libllama-server-impl.so|libllama-bench-impl.so|libllama-perplexity-impl.so) ;;
        *) rm -f "$f" ;; esac
done
n_variants=$(ls "${STAGE}${BONSAI_LIBDIR}"/libggml-cpu-*.so | wc -l)
[ "${n_variants}" -ge 2 ] || die "CPU backend variants missing (${n_variants})"
# OpenMP-threaded variant modules (same kernels, v0.2 threading) for A/B comparisons and fallback
if [ "${REQUIRE_OMP_VARIANT:-1}" = 1 ] || ls "${BUILD}"/bin/omp/libggml-cpu-*.so >/dev/null 2>&1; then
    n_omp=0
    for f in "${BUILD}"/bin/omp/libggml-cpu-*.so; do install -D -m 0644 "$f" "${STAGE}${BONSAI_LIBDIR}/omp/$(basename "$f")"; n_omp=$((n_omp + 1)); done
    [ "${n_omp}" -eq "${n_variants}" ] || die "OpenMP variant modules: ${n_omp}, expected ${n_variants}"
fi

# strip, keep build-id
find "${STAGE}/usr" -type f \( -name '*.so*' -o -perm -u+x \) -exec sh -c 'file -b "$1" | grep -q ELF && strip --strip-unneeded --remove-section=.comment --remove-section=.note "$1"' _ {} \;

# point every ELF at the private library directory (DT_RPATH: wins over LD_LIBRARY_PATH, so another
# llama.cpp build on LD_LIBRARY_PATH can never be mixed in) and verify nothing references the build tree
command -v patchelf >/dev/null || die "patchelf is required"
while IFS= read -r -d '' f; do
    file -b "$f" | grep -q ELF || continue
    patchelf --force-rpath --set-rpath "${BONSAI_LIBDIR}" "$f"
done < <(find "${STAGE}/usr" -type f -print0)
if grep -rl --binary-files=binary "${BUILD}" "${STAGE}/usr" >/dev/null 2>&1; then
    die "build-tree path leaked into: $(grep -rl "${BUILD}" "${STAGE}/usr" | head -3)"
fi
# every NEEDED library must be either private (shipped here) or a system library
for f in "${STAGE}"/usr/bin/bonsai-* "${STAGE}${BONSAI_LIBDIR}"/bin/* "${STAGE}${BONSAI_LIBDIR}"/*.so* "${STAGE}${BONSAI_LIBDIR}"/omp/*.so; do
    [ -e "$f" ] || continue
    file -b "$f" | grep -q ELF || continue
    for n in $(readelf -d "$f" | sed -n 's/.*(NEEDED).*\[\(.*\)\]/\1/p'); do
        case "$n" in libllama*|libggml*|libmtmd*) [ -e "${STAGE}${BONSAI_LIBDIR}/$n" ] || die "$(basename "$f") needs $n, not packaged";; esac
    done
done

# documentation and licences
DOC="${STAGE}/usr/share/doc/${PKG}"
install -m 0644 "${REPO_ROOT}/packaging/README.Debian.md" "${DOC}/README.md"
install -m 0644 "${REPO_ROOT}/CHANGELOG.md" "${DOC}/CHANGELOG.md"
gzip -9n -c "${REPO_ROOT}/packaging/changelog.Debian" > "${DOC}/changelog.Debian.gz"
install -m 0644 "${SRC_DIR}/BONSAI_PROVENANCE.txt" "${DOC}/PROVENANCE.txt"
install -m 0644 "${REPO_ROOT}/packaging/copyright" "${DOC}/copyright"
mkdir -p "${DOC}/licenses"
install -m 0644 "${SRC_DIR}/LICENSE" "${DOC}/licenses/llama.cpp-LICENSE"
install -m 0644 "${SRC_DIR}/AUTHORS" "${DOC}/licenses/llama.cpp-AUTHORS"
for l in "${SRC_DIR}"/licenses/*; do install -m 0644 "$l" "${DOC}/licenses/$(basename "$l")"; done
install -m 0644 "${SRC_DIR}/vendor/cpp-httplib/LICENSE"        "${DOC}/licenses/cpp-httplib-LICENSE"
install -m 0644 "${SRC_DIR}/vendor/hash/xxhash/LICENSE"        "${DOC}/licenses/xxhash-LICENSE"
install -m 0644 "${SRC_DIR}/vendor/hash/rotate-bits/LICENSE.md" "${DOC}/licenses/rotate-bits-LICENSE.md"
install -m 0644 "${REPO_ROOT}/LICENSE" "${DOC}/licenses/bonsai-oryon-LICENSE"

# dependencies from the ELF files (private libs resolved inside the package)
mkdir -p "${BUILD_ROOT}/shlibs/debian"
printf 'Source: %s\n\nPackage: %s\nArchitecture: %s\n' "${PKG}" "${PKG}" "${ARCH}" > "${BUILD_ROOT}/shlibs/debian/control"
DEPS=$(cd "${BUILD_ROOT}/shlibs" && dpkg-shlibdeps -O -l"${STAGE}${BONSAI_LIBDIR}" \
          --ignore-missing-info \
          $(for f in "${STAGE}"/usr/bin/bonsai-* "${STAGE}${BONSAI_LIBDIR}"/bin/* "${STAGE}${BONSAI_LIBDIR}"/*.so* "${STAGE}${BONSAI_LIBDIR}"/omp/*.so; do [ -e "$f" ] && file -b "$f" | grep -q ELF && echo "$f"; done) 2>/dev/null | sed -n 's/^shlibs:Depends=//p')
[ -n "${DEPS}" ] || die "dpkg-shlibdeps produced no dependencies"
log "Depends: ${DEPS}"

SIZE=$(du -sk --exclude=DEBIAN "${STAGE}" | cut -f1)
sed -e "s|@VERSION@|${BONSAI_VERSION}|" -e "s|@ARCH@|${ARCH}|" -e "s|@DEPENDS@|${DEPS}|" -e "s|@SIZE@|${SIZE}|" \
    "${REPO_ROOT}/packaging/control.in" > "${STAGE}/DEBIAN/control"
(cd "${STAGE}" && find usr -type f -exec md5sum {} + | sort -k2) > "${STAGE}/DEBIAN/md5sums"

find "${STAGE}" -type d -exec chmod 0755 {} +
dpkg-deb --root-owner-group -Zxz --build "${STAGE}" "${DEB}" >/dev/null
(cd "${OUTDIR}" && sha256sum "$(basename "${DEB}")" > "$(basename "${DEB}").sha256")
log "package: ${DEB}"
dpkg-deb --info "${DEB}" >&2
