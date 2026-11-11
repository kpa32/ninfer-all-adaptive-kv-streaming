# Manual Engine drivers for the KV-streaming checks

Two standalone C++ harnesses that drive the public `Engine` where the automated suites cannot reach:
the test tree is an aggregate binary (a full rebuild for one case), and these checks need either two
`Engine` instances in one process or a controlled pool geometry.

They are **manual**: nothing in CI builds or runs them. Build them against the tree's own link line so
they always match the current build.

```bash
cd <repo>/build-sm89
LIBS=$(ninja -t commands apps/ninfer | tail -1 | sed 's/^.*-o apps\/ninfer //')
/usr/bin/g++ -std=c++20 -O2 -I../include -I../src -I/usr/local/cuda/include \
  -c ../tools/bench/kv_stream/geometry_driver.cpp -o /tmp/geometry_driver.o
eval "/usr/bin/g++ -O3 -DNDEBUG /tmp/geometry_driver.o -o /tmp/geometry_driver $LIBS"
```

(Relink after every rebuild of the model/runtime libraries — a driver linked before a fix is a stale
oracle. That mistake produced a false "still diverging" reading once.)

## `geometry_driver.cpp` — the same prompt must answer the same whatever the pool's shape

Two streamed Engines in one process, same build, same prompt, same weights. The only difference is how
much device memory is reserved before each engine is created, which changes what `--kv-capacity auto`
measures and therefore the pool: window depth per layer and ring depth.

```bash
# equal pools (control)
/tmp/geometry_driver <artifact.ninfer>
# pools that differ by 64 MiB / 512 MiB / 1 GiB
LP_RESERVE2=64   /tmp/geometry_driver <artifact.ninfer>
LP_RESERVE2=512  /tmp/geometry_driver <artifact.ninfer>
LP_RESERVE2=1024 /tmp/geometry_driver <artifact.ninfer>
# the second arm keeps its context (context cache, host StateImage, prefix reuse)
LP_ARM2_RETAINED=1 /tmp/geometry_driver <artifact.ninfer>
```

Expected: `resident`/`ring` differ between the arms while `ids equal=1` and `logprobs equal=1`, and
the exit code is 0. A failure means the streamed shape reached the arithmetic, which is the defect class recorded
in `bench/results/kv-stream/layout-rename-defect.md`.

Knobs: `LP_RESERVE1`/`LP_RESERVE2` (MiB), `LP_PROMPT_TOKENS` (default: the whole filler prompt,
4607 tokens), `LP_CHUNK` (default 256), `LP_MAX_CONTEXT` (default 262144), `LP_NO_A8`, `LP_NO_SPEC`,
`LP_HOST_SLOTS`, `LP_COLD_TWICE`, `LP_SKIP_COLD`, `LP_SKIP_RETAINED`.

## `reuse_driver.cpp` — a finished request's context is reused by the next one

Two requests on one Engine: a prompt plus the tokens it generated. The second request must reuse the
first one's retained frontier, prefill only the increment, and produce exactly the tokens a cache-less
run of the same prompt produces.

```bash
/tmp/reuse_driver <artifact.ninfer>
```

Expected: `first_equal=1`, `retained=1 frontier=<prompt+generated-1>`, `reused=<same>`, `equal_ids=1`,
`OK`, exit code 0.
