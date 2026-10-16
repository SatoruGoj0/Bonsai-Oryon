#!/usr/bin/env bash
# Functional tests for the bonsai-oryon binaries (CLI, bench, HTTP server, CPU variant selection).
#   run-functional.sh <bindir> <model.gguf> [results.md]
# <bindir> holds bonsai-cli/bonsai-server/bonsai-bench (installed package: /usr/bin) or the
# llama-* names of a build tree (then set BONSAI_BACKEND_DIR to the build's bin dir).
set -uo pipefail
BIN="$1"; MODEL="$(realpath "$2")"; RESULTS="${3:-/dev/null}"
HERE="$(cd "$(dirname "$0")" && pwd)"
pick() { if [ -x "$BIN/bonsai-$1" ]; then echo "$BIN/bonsai-$1"; else echo "$BIN/llama-$1"; fi; }
CLI=$(pick cli); SERVER=$(pick server); BENCH=$(pick bench)
T=${THREADS:-2}
EMU=${EMU:-}   # e.g. "qemu-aarch64 -cpu max" for local cross testing
WORK=$(mktemp -d); SPID=""
trap '[ -n "$SPID" ] && kill "$SPID" 2>/dev/null; rm -rf "$WORK"' EXIT
pass=0; fail=0
ok()   { echo "PASS  $1"; echo "| $1 | PASS | ${2:-} |" >> "$RESULTS"; pass=$((pass+1)); }
bad()  { echo "FAIL  $1: ${2:-}"; echo "| $1 | **FAIL** | ${2:-} |" >> "$RESULTS"; fail=$((fail+1)); }

echo "| test | result | detail |" >> "$RESULTS"; echo "|---|---|---|" >> "$RESULTS"

# 1. CLI single-turn generation (greedy)
timeout 600 $EMU "$CLI" -m "$MODEL" -t "$T" -ngl 0 -c 512 -n 24 --temp 0 --seed 1 -st -p "hello world" \
    < /dev/null > "$WORK/cli.out" 2> "$WORK/cli.err"
rc=$?
if [ $rc -eq 0 ] && [ -s "$WORK/cli.out" ]; then ok "cli-generate" "exit 0, $(wc -c < "$WORK/cli.out") bytes of output"
else bad "cli-generate" "exit $rc; $(tail -5 "$WORK/cli.err" | tr '\n' ' ')"; fi

# 2. runtime CPU detection: the loaded CPU module must be the best one this CPU supports
#    (llama-bench logs "load_backend: loaded CPU backend from <path>")
timeout 600 $EMU "$BENCH" -m "$MODEL" -t "$T" -ngl 0 -p 8 -n 2 -r 1 > /dev/null 2> "$WORK/load.err"
want=${EXPECT_VARIANT:-$("$HERE/expected-cpu-variant.sh")}
got=$(grep -oE 'loaded CPU backend from [^ ]+' "$WORK/load.err" | head -1 | grep -oE 'libggml-cpu-[A-Za-z0-9_.]+\.so' | sed 's/^libggml-cpu-//; s/\.so$//')
if [ -n "$got" ] && [ "$got" = "$want" ]; then ok "cpu-variant-selection" "loaded $got (expected ${EXPECT_VARIANT:+[env] }from CPU features: $want)"
else bad "cpu-variant-selection" "loaded '${got}', expected '$want'"; grep -i backend "$WORK/load.err" | head; fi
case "$(grep -oE 'loaded CPU backend from [^ ]+' "$WORK/load.err" | head -1)" in
    *"${EXPECT_BACKEND_DIR:-/usr/lib/bonsai-oryon}"/*) ok "backend-dir" "$(grep -oE 'loaded CPU backend from [^ ]+' "$WORK/load.err" | head -1)";;
    *) bad "backend-dir" "$(grep -i 'load_backend' "$WORK/load.err" | head -3 | tr '\n' ' ')";;
esac
$EMU "$CLI" --version > "$WORK/ver.txt" 2>&1
grep -q "bonsai-oryon" "$WORK/ver.txt" && ok "version-string" "$(grep -m1 -i version "$WORK/ver.txt")" || bad "version-string" "$(head -3 "$WORK/ver.txt")"

# 3. no foreign backend may be picked up from the current directory
mkdir -p "$WORK/trap" && printf 'not an elf' > "$WORK/trap/libggml-vulkan-x.so" && printf 'junk' > "$WORK/trap/libggml-cpu-zzz.so"
(cd "$WORK/trap" && timeout 600 $EMU "$BENCH" -m "$MODEL" -t "$T" -ngl 0 -p 8 -n 2 -r 1 > /dev/null 2> "$WORK/trap.err")
if grep -q "$WORK/trap" "$WORK/trap.err" || ! grep -q "loaded CPU backend" "$WORK/trap.err"; then bad "backend-search-isolation" "$(grep -i load_backend "$WORK/trap.err" | head -3 | tr '\n' ' ')"
else ok "backend-search-isolation" "stray libggml-*.so in CWD ignored"; fi
if grep -iE 'load_backend: loaded' "$WORK/load.err" | grep -qviE 'loaded CPU backend'; then bad "cpu-only" "$(grep load_backend "$WORK/load.err")"; else ok "cpu-only" "only the CPU backend was loaded"; fi

# 4. bench: pp/tg with JSON output
if timeout 600 $EMU "$BENCH" -m "$MODEL" -t "$T" -ngl 0 -p 64 -n 16 -r 2 -o json > "$WORK/bench.json" 2> "$WORK/bench.err" \
   && python3 -c "import json,sys; d=json.load(open(sys.argv[1])); assert len(d)==2 and all(r['avg_ts']>0 for r in d); print(' '.join('%s=%.1f' % ('pp' if r['n_prompt'] else 'tg', r['avg_ts']) for r in d))" "$WORK/bench.json" > "$WORK/bench.sum"; then
    ok "bench-json" "tok/s on CI runner (tiny synthetic model): $(cat "$WORK/bench.sum")"
else bad "bench-json" "$(tail -3 "$WORK/bench.err" | tr '\n' ' ')"; fi

# 5. HTTP server, OpenAI-compatible API
PORT=${PORT:-18080}
$EMU "$SERVER" -m "$MODEL" --alias bonsai-9b --port "$PORT" -t "$T" -ngl 0 -c 1024 > "$WORK/srv.log" 2>&1 &
SPID=$!
for i in $(seq 1 600); do curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && break; sleep 0.5; done
if curl -sf "http://127.0.0.1:$PORT/health" | grep -q ok; then ok "server-health" "/health ok"; else bad "server-health" "$(tail -5 "$WORK/srv.log" | tr '\n' ' ')"; fi

# default bind address must be loopback only
listen=$(python3 - "$PORT" <<'PY'
import sys, socket, struct
port = int(sys.argv[1]); out = []
for f, fam in (("/proc/net/tcp", 4), ("/proc/net/tcp6", 6)):
    try: lines = open(f).read().splitlines()[1:]
    except OSError: continue
    for l in lines:
        local, state = l.split()[1], l.split()[3]
        a, p = local.split(":")
        if state != "0A" or int(p, 16) != port: continue   # 0A = LISTEN
        if fam == 4: out.append(socket.inet_ntoa(struct.pack("<I", int(a, 16))))
        else: out.append(socket.inet_ntop(socket.AF_INET6, b"".join(struct.pack("<I", int(a[i:i+8], 16)) for i in range(0, 32, 8))))
print(" ".join(out))
PY
)
case "$listen" in 127.0.0.1|::1) ok "server-default-bind-loopback" "listening only on $listen (no --host given)";;
     *) bad "server-default-bind-loopback" "listening on '${listen}'";; esac

curl -sf "http://127.0.0.1:$PORT/v1/models" > "$WORK/models.json"
python3 -c "import json,sys; d=json.load(open(sys.argv[1])); ids=[m['id'] for m in d['data']]; assert 'bonsai-9b' in ids, ids" "$WORK/models.json" \
    && ok "api-v1-models" "alias bonsai-9b listed" || bad "api-v1-models" "$(head -c 300 "$WORK/models.json")"

curl -sf "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"bonsai-9b","messages":[{"role":"system","content":"You are terse."},{"role":"user","content":"Say hello"}],"max_tokens":16,"temperature":0}' > "$WORK/chat.json"
python3 -c "
import json,sys; d=json.load(open(sys.argv[1]))
c=d['choices'][0]; assert c['message']['role']=='assistant'; assert isinstance(c['message']['content'], str)
assert d['object']=='chat.completion' and d['usage']['completion_tokens']>0
print(d['usage'])" "$WORK/chat.json" > "$WORK/chat.sum" 2>&1 \
    && ok "api-chat-completions" "$(cat "$WORK/chat.sum")" || bad "api-chat-completions" "$(head -c 300 "$WORK/chat.json") $(cat "$WORK/chat.sum")"

curl -sfN "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"bonsai-9b","messages":[{"role":"user","content":"Count"}],"max_tokens":8,"stream":true}' > "$WORK/stream.txt"
if grep -q '^data: {' "$WORK/stream.txt" && grep -q '^data: \[DONE\]' "$WORK/stream.txt"; then ok "api-chat-streaming" "$(grep -c '^data: {' "$WORK/stream.txt") SSE chunks + [DONE]"
else bad "api-chat-streaming" "$(head -c 300 "$WORK/stream.txt")"; fi

curl -sf "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
    -d '{"model":"bonsai-9b","prompt":"The tree","max_tokens":8,"temperature":0}' > "$WORK/compl.json"
python3 -c "import json,sys; d=json.load(open(sys.argv[1])); assert isinstance(d['choices'][0]['text'], str)" "$WORK/compl.json" \
    && ok "api-completions" "text completion ok" || bad "api-completions" "$(head -c 300 "$WORK/compl.json")"

# tool-call style request (used by OpenCode): must be accepted and answered with valid JSON
curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"bonsai-9b","messages":[{"role":"user","content":"weather?"}],"max_tokens":8,"tools":[{"type":"function","function":{"name":"get_weather","description":"x","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]}' > "$WORK/tools.json"
python3 -c "import json,sys; d=json.load(open(sys.argv[1])); assert 'choices' in d or 'error' in d" "$WORK/tools.json" \
    && ok "api-tools-request-shape" "$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print('choices' if 'choices' in d else 'error: '+d['error'].get('message','')[:120])" "$WORK/tools.json")" \
    || bad "api-tools-request-shape" "$(head -c 300 "$WORK/tools.json")"

kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null; SPID=""

echo
echo "functional: $pass passed, $fail failed"
echo "" >> "$RESULTS"; echo "**functional: $pass passed, $fail failed**" >> "$RESULTS"
[ $fail -eq 0 ]
