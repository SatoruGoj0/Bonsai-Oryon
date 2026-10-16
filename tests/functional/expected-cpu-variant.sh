#!/usr/bin/env bash
# Print the libggml-cpu-<variant> the loader SHOULD pick on this machine, computed independently
# from /proc/cpuinfo with the same scoring rules as ggml's cpu-feats.cpp (Linux variant list +
# Bonsai-Oryon's armv8.6_nosve). Used to verify runtime CPU feature detection end to end.
set -euo pipefail
feats=" $(grep -m1 -i '^Features' /proc/cpuinfo | cut -d: -f2) "
has() { [[ "$feats" == *" $1 "* ]]; }
dot=0; fp16=0; sve=0; i8mm=0; sve2=0; sme=0
has asimddp && dot=1
has asimdhp && fp16=1
has sve && sve=1
has i8mm && i8mm=1
has sve2 && sve2=1
has sme && sme=1
best=armv8.0_1; bestscore=1
try() { # name dot fp16 sve i8mm sve2 sme
    local n=$1 s=1
    [ "$2" = 1 ] && { [ $dot = 1 ] || return 0; s=$((s+2)); }
    [ "$3" = 1 ] && { [ $fp16 = 1 ] || return 0; s=$((s+4)); }
    [ "$4" = 1 ] && { [ $sve = 1 ] || return 0; s=$((s+8)); }
    [ "$5" = 1 ] && { [ $i8mm = 1 ] || return 0; s=$((s+16)); }
    [ "$6" = 1 ] && { [ $sve2 = 1 ] || return 0; s=$((s+32)); }
    [ "$7" = 1 ] && { [ $sme = 1 ] || return 0; s=$((s+64)); }
    if [ $s -gt $bestscore ]; then best=$n; bestscore=$s; fi
}
try armv8.2_1     1 0 0 0 0 0
try armv8.2_2     1 1 0 0 0 0
try armv8.2_3     1 1 1 0 0 0
try armv8.6_1     1 1 1 1 0 0
try armv8.6_nosve 1 1 0 1 0 0
try armv8.6_2     1 1 1 1 1 0
try armv9.2_1     1 1 1 1 0 1
try armv9.2_2     1 1 1 1 1 1
echo "$best"
