#!/usr/bin/env bash
# Run a tool from an older Bonsai Oryon release .deb without installing it (and without touching the installed
# package): the .deb is extracted to ~/.cache/bonsai-oryon/releases/<version>/ and the binary is started through
# the dynamic loader with its built-in RPATH (/usr/lib/bonsai-oryon) disabled and the extracted libraries used
# instead. No root needed.
#
#   bonsai-release-run.sh OLD.deb [--tool bonsai-bench|bonsai-cli|bonsai-server] [ARGS...]
#
# Example: bonsai-release-run.sh bonsai-oryon_0.2.0_arm64.deb -m model.gguf -t 4 -p 256 -n 128
set -euo pipefail
deb="${1:?usage: $0 OLD.deb [--tool NAME] [ARGS...]}"; shift
tool=bonsai-bench
if [ "${1:-}" = --tool ]; then tool="$2"; shift 2; fi
ver=$(dpkg-deb -f "$deb" Version)
dir="${XDG_CACHE_HOME:-$HOME/.cache}/bonsai-oryon/releases/$ver"
if [ ! -x "$dir/usr/bin/$tool" ]; then
    mkdir -p "$dir.tmp"
    dpkg-deb -x "$deb" "$dir.tmp"
    rm -rf "$dir"; mv "$dir.tmp" "$dir"
fi
lib="$dir/usr/lib/bonsai-oryon"
ldso=$(ls /lib/ld-linux-aarch64.so.1 /lib64/ld-linux-x86-64.so.2 2>/dev/null | head -1)
[ -n "$ldso" ] || { echo "dynamic loader not found" >&2; exit 1; }
# every object of the release whose RPATH must be ignored (glibc matches the loaded path)
inhibit="$dir/usr/bin/$tool"
while IFS= read -r f; do inhibit="$inhibit:$f"; done < <(find "$lib" -name '*.so*' -type f)
export BONSAI_BACKEND_DIR="$lib"
exec "$ldso" --inhibit-rpath "$inhibit" --library-path "$lib" "$dir/usr/bin/$tool" "$@"
