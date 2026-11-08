#!/usr/bin/env bash
# Records or checks the generated token ids of the streaming equivalence cases
# (docs/maintainer/kv-stream-golden/MANIFEST.md).
#
# The check this exists for: a streamed KV cache must answer exactly like an all-resident one.
# Both configurations run the same binary and the same prompt; the ids of every case are kept in
# this repository, so a later change is judged by comparing against them instead of re-running the
# other side of the comparison every time.
#
#   scripts/kv-stream-golden.sh --record [case...]   # write (or overwrite) the recorded ids
#   scripts/kv-stream-golden.sh [case...]            # compare against the recorded ids
#
# Cases pin every flag that moves the arithmetic (--prefill-chunk included: chunking changes the
# batched shapes and therefore the tokens), so a comparison never mixes two different questions.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=${NINFER_BIN:-$ROOT/build-sm89/apps/ninfer}
MODEL=${NINFER_KV_STREAM_MODEL:-$HOME/ninfer-fork/models/Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer}
GOLDEN=$ROOT/docs/maintainer/kv-stream-golden
WORK=${TMPDIR:-/tmp}/kv-stream-golden

record=0
if [[ ${1:-} == --record ]]; then
    record=1
    shift
fi

if [[ ! -x $BIN ]]; then
    echo "error: no binary at $BIN (build build-sm89, or set NINFER_BIN)" >&2
    exit 2
fi
if [[ ! -f $MODEL ]]; then
    echo "error: no model at $MODEL (set NINFER_KV_STREAM_MODEL)" >&2
    exit 2
fi

mkdir -p "$WORK" "$GOLDEN"

# A prompt of `words` repeated sentences plus a sharply predictable tail: the recorded ids are only
# a fair oracle where the model's next token is not a near tie.
prompt() {
    local words=$1 path=$2
    python3 - "$words" "$path" <<'PY'
import json, sys
words, path = int(sys.argv[1]), sys.argv[2]
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

# case: name words capacity extra flags...
case_words() { case $1 in equivalence-190k) echo 184000 ;; streamed-246k) echo 246600 ;; esac; }
case_capacity() { case $1 in equivalence-190k) echo 190016 ;; streamed-246k) echo 262144 ;; esac; }
case_extra() {
    case $1 in
        # All resident: the pool holds the whole context, so streaming takes no part.
        equivalence-190k) echo "--kv-headroom-mib 8" ;;
        equivalence-190k-streamed) echo "--kv-stream" ;;
        # Streamed: the same prompt, but the pool cannot hold it and every layer streams.
        streamed-246k) echo "--kv-stream" ;;
        *) return 1 ;;
    esac
}

cases=("$@")
if [[ ${#cases[@]} -eq 0 ]]; then
    cases=(equivalence-190k equivalence-190k-streamed streamed-246k)
fi

for name in "${cases[@]}"; do
    extra=$(case_extra "$name") || { echo "error: unknown case $name" >&2; exit 2; }
    words=$(case_words "$name")
    capacity=$(case_capacity "$name")
    prompt "$words" "$WORK/$words.json"
    log=$WORK/$name.log
    echo "=== $name (capacity $capacity, $words words) $extra"
    NINFER_KV_STREAM_REPORT=${NINFER_KV_STREAM_REPORT:-0} \
    "$BIN" "$MODEL" --messages "$WORK/$words.json" --max-context "$capacity" \
        --kv-capacity auto --kv-dtype rk4v4 --gdn-state-fp16 --prefill-chunk 256 \
        --greedy --seed 42 --invariant-attention --max-new 16 \
        --spec mtp --draft-tokens 3 --print-token-ids $extra >/dev/null 2>"$log"
    ids=$(grep -o 'generated ids.*' "$log" | head -1 | sed 's/generated ids *//')
    if [[ -z $ids ]]; then
        echo "error: $name produced no token ids; see $log" >&2
        exit 1
    fi
    echo "$ids" >"$WORK/$name.ids"
    grep -E 'prompt tokens|decode speed|stream window|stream ring' "$log" | sed 's/^summary *//' | sed 's/^/    /'
    if ((record)); then
        cp "$WORK/$name.ids" "$GOLDEN/$name.ids"
        echo "    recorded $GOLDEN/$name.ids"
    elif [[ -f $GOLDEN/$name.ids ]]; then
        if diff -q "$GOLDEN/$name.ids" "$WORK/$name.ids" >/dev/null; then
            echo "    MATCHES the recorded ids"
        else
            echo "    DIFFERS from the recorded ids:"
            diff <(tr ' ' '\n' <"$GOLDEN/$name.ids") <(tr ' ' '\n' <"$WORK/$name.ids") | head -20 | sed 's/^/      /'
        fi
    else
        echo "    no recorded ids for $name (run with --record)"
    fi
done

# The equivalence itself: the same prompt answered by an all-resident and a streamed cache.
if [[ -f $WORK/equivalence-190k.ids && -f $WORK/equivalence-190k-streamed.ids ]]; then
    if diff -q "$WORK/equivalence-190k.ids" "$WORK/equivalence-190k-streamed.ids" >/dev/null; then
        echo "=== equivalence-190k: all-resident and streamed ids are identical"
    else
        echo "=== equivalence-190k: all-resident and streamed ids DIFFER"
        diff <(tr ' ' '\n' <"$WORK/equivalence-190k.ids") \
             <(tr ' ' '\n' <"$WORK/equivalence-190k-streamed.ids") | head -20
    fi
fi
