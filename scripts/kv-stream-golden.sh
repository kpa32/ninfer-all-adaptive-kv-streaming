#!/usr/bin/env bash
# The streaming equivalence check: a streamed KV cache must answer exactly like the original
# NInfer binary that has no streaming at all.
#
# The oracle is the *original* build - `/home/kpa/ninfer/ninfer-all/build-sm89/apps/ninfer`, the
# user's own prebuilt binary. It is borrowed read-only for test runs: never built, never written,
# never configured (`--record` only *executes* it). Its token ids are recorded in
# docs/maintainer/kv-stream-golden/, so a later change is judged against them instead of re-running
# the other side of the comparison every time.
#
#   scripts/kv-stream-golden.sh --record [case...]   # record the ORIGINAL binary's ids
#   scripts/kv-stream-golden.sh [case...]            # run this build, compare with the records
#
# Every flag that moves the arithmetic is pinned, `--prefill-chunk` included: chunking changes the
# batched shapes and was measured to move the tokens (measurements M-08), so a comparison never
# mixes two questions.
#
# The cases are the contexts both layouts can serve. Above roughly 186K the all-resident layout
# with MTP does not start on this card, and below the streamed window (181 440 tokens at
# `rk4v4` + MTP3) streaming takes no part - 184K is the range where one is possible and the other
# engaged, which is why it is the case that matters.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OURS=${NINFER_BIN:-$ROOT/build-sm89/apps/ninfer}
BASE=${NINFER_BASE_BIN:-/home/kpa/ninfer/ninfer-all/build-sm89/apps/ninfer}
MODEL=${NINFER_KV_STREAM_MODEL:-$HOME/ninfer-fork/models/Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer}
GOLDEN=$ROOT/docs/maintainer/kv-stream-golden
WORK=${TMPDIR:-/tmp}/kv-stream-golden
STREAM_FLOOR=$((184000))   # words; every case at or above this streams in the --kv-stream run

record=0
if [[ ${1:-} == --record ]]; then
    record=1
    shift
fi

if [[ $record == 1 && ! -x $BASE ]]; then
    echo "error: the original binary is not at $BASE (the oracle; set NINFER_BASE_BIN)" >&2
    exit 2
fi
if [[ $record == 0 && ! -x $OURS ]]; then
    echo "error: no build to compare at $OURS (build build-sm89, or set NINFER_BIN)" >&2
    exit 2
fi
if [[ ! -f $MODEL ]]; then
    echo "error: no model at $MODEL (set NINFER_KV_STREAM_MODEL)" >&2
    exit 2
fi

mkdir -p "$WORK" ${record:+"$GOLDEN"}

# A prompt built from varied sentences plus a sharply predictable tail: recorded ids are only a fair
# oracle where the model's next token is not a near tie.
prompt() {
    local words=$1 path=$2
    python3 - "$words" "$path" <<'PY'
import json, sys
words, path = int(sys.argv[1]), int(sys.argv[2])
sentence = ("NInfer keeps every KV page of the context in pinned host memory and streams the newest "
            "window onto the device, so a round copies only the share the window lacks. ")
body, count = [], 0
while count < words:
    body.append(sentence)
    count += len(sentence.split())
text = "".join(body)
text += "\n\nThe capital of France is"
json.dump([{"role": "user", "content": text}], open(path, "w"))
PY
}

case_words()    { case $1 in 8k) echo 8000 ;; 100k) echo 100000 ;; 184k) echo 184000 ;; esac; }
case_capacity() { case $1 in 8k) echo 8192 ;; 100k) echo 100000 ;; 184k) echo 184064 ;; esac; }

# Runs one configuration and writes its ids to $WORK/$tag.ids.
run_case() {
    local tag=$1 binary=$2 words=$3 capacity=$4
    shift 4
    local log=$WORK/$tag.log
    echo "=== $tag: $(basename "$(dirname "$(dirname "$binary")")") capacity $capacity, $words words, $*"
    "$binary" "$MODEL" --messages "$WORK/$words.json" --max-context "$capacity" \
        --kv-capacity auto --kv-dtype rk4v4 --gdn-state-fp16 --prefill-chunk 256 \
        --greedy --seed 42 --invariant-attention --max-new 16 \
        --spec mtp --draft-tokens 3 --print-token-ids "$@" >/dev/null 2>"$log"
    local rc=$?
    local ids
    ids=$(grep -o 'generated ids.*' "$log" | head -1 | sed 's/generated ids *//')
    if [[ $rc != 0 || -z $ids ]]; then
        echo "    FAILED (exit $rc); last line: $(tail -1 "$log")"
        return 1
    fi
    printf '%s\n' "$ids" >"$WORK/$tag.ids"
    grep -E 'prompt tokens|decode speed|page groups|stream window|stream ring' "$log" |
        sed 's/^summary *//' | sed 's/^/    /'
    return 0
}

cases=("$@")
if [[ ${#cases[@]} -eq 0 ]]; then
    cases=(8k 100k 184k)
fi

failures=0
for name in "${cases[@]}"; do
    words=$(case_words "$name") || { echo "error: unknown case $name" >&2; exit 2; }
    capacity=$(case_capacity "$name")
    prompt "$words" "$WORK/$words.json"

    if ((record)); then
        # The oracle: the original binary, all resident, exactly as the user built it.
        run_case "base-$name" "$BASE" "$words" "$capacity" || { failures=$((failures + 1)); continue; }
        cp "$WORK/base-$name.ids" "$GOLDEN/$name.ids"
        echo "    recorded $GOLDEN/$name.ids"
        continue
    fi

    if [[ ! -f $GOLDEN/$name.ids ]]; then
        echo "=== $name: no recorded ids (run --record against the original first)"
        failures=$((failures + 1))
        continue
    fi
    # Our build, the same configuration, and - where the context is past the window - the streamed
    # layout as well. Both must answer like the original.
    variants=(resident)
    (( words >= STREAM_FLOOR )) && variants+=(streamed)
    for variant in "${variants[@]}"; do
        extra=()
        [[ $variant == streamed ]] && extra=(--kv-stream)
        if run_case "$name-$variant" "$OURS" "$words" "$capacity" "${extra[@]}"; then
            if diff -q "$GOLDEN/$name.ids" "$WORK/$name-$variant.ids" >/dev/null; then
                echo "    $variant: MATCHES the original's ids"
            else
                echo "    $variant: DIFFERS from the original's ids"
                diff <(tr ' ' '\n' <"$GOLDEN/$name.ids") <(tr ' ' '\n' <"$WORK/$name-$variant.ids") |
                    head -10 | sed 's/^/      /'
                failures=$((failures + 1))
            fi
        else
            failures=$((failures + 1))
        fi
    done
done

if ((failures != 0)); then
    echo "=== $failures case(s) failed or differed"
    exit 1
fi
echo "=== every case answered like the original binary"
