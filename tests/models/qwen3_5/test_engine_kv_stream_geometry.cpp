// C14: the same prompt answers the same whatever its pool's shape.
//
// A streamed page's device slot is its layer's base plus its position in the resident window, so a
// plan that names pages differently puts another page's KV behind the same attention. The answer may
// not see that: the layout is planned once, when the Engine starts, and the arithmetic is the same
// arithmetic. This is the cell the equivalence suite could not see - every other check compares two
// runs of the *same* capacity, so both arms were planned the same way and the defect
// (bench/results/kv-stream/layout-rename-defect.md) was invisible by construction.
//
// Two Engines of this build, one process, the same prompt and the same weights, with capacities that
// plan different windows and different ring depths. Every token id and every log probability must
// match bit for bit, and the two geometries must actually differ - a pair that planned the same pool
// proves nothing, so it skips rather than passing.
//
// The pool is what a run's capacity grants it (auto or explicit), so this varies the capacity rather
// than the free memory: a reservation would make the same point on one host and silently stop
// differing on a bigger card.
//
// Set NINFER_TEST_ARTIFACT to a Qwen3.8-27B v3 artifact. Without it the test skips with 77.

#include "guarded_main.h"

#include "ninfer/engine.h"
#include "ninfer/kv_cache_storage.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using ninfer::Engine;
using ninfer::EngineOptions;
using ninfer::EnginePurpose;
using ninfer::KvCacheStorage;
using ninfer::KvCapacityPolicy;
using ninfer::RequestOptions;
using ninfer::StreamedKvStats;
using ninfer::TokenId;

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

// One sentence repeated past the window an explicit capacity leaves, so both arms stream. The text
// is filler on purpose: the check is that the two arms agree, not what they say.
const char* kFiller =
    "NInfer keeps every KV page of the context in pinned host memory and streams the newest window "
    "onto the device. ";
constexpr int kSentenceRepeats = 30;

EngineOptions engine_options(const char* artifact, std::uint32_t capacity_tokens) {
    EngineOptions options;
    options.artifact_path   = artifact;
    options.purpose         = EnginePurpose::Generation;
    options.max_context     = 262144;
    options.kv_capacity     = KvCapacityPolicy::explicit_capacity(capacity_tokens);
    options.kv_stream       = true;
    options.kv_cache        = KvCacheStorage::RotatedLloyd4KeyInt4Value;
    options.prefill_chunk   = 256;
    options.max_concurrency = 1;
    options.prefill_a8      = true;
    // The report is comparable across the arms only if the arithmetic is: the same rule for every
    // key segment on both sides (docs/cli.md, --invariant-attention).
    options.invariant_attention = true;
    // The streamed shape: the pinned host authority is the KV tier, so the cache itself takes no
    // host tier, no disk tier and no slot (kv_stream_guards.h rejects every one of them beside a
    // streamed pool).
    options.context_cache.enabled                           = false;
    options.context_cache.mode                              = ninfer::ContextCacheMode::Legacy;
    options.context_cache.host_state_slots                  = 0;
    options.context_cache.host_kv_capacity_bytes            = 0;
    options.context_cache.device_state_slots                = 0;
    options.context_cache.max_private_continuations         = 1;
    options.context_cache.max_shared_prefixes               = 0;
    options.context_cache.max_long_anchors_per_continuation = 0;
    options.context_cache.host_cache_budget_bytes.reset();
    options.context_cache.disk_kv_path.clear();
    return options;
}

RequestOptions greedy() {
    RequestOptions out;
    out.execution.requested_output_tokens = 8;
    out.execution.sampling.temperature     = 0;
    out.execution.allow_prefix_reuse       = false;
    out.execution.logprobs                 = true;
    out.stop.include_model_defaults        = false;
    return out;
}

struct Arm {
    std::vector<TokenId> ids;
    std::vector<float> logprobs;
    StreamedKvStats stats;
};

Arm answer(const char* artifact, std::uint32_t capacity_tokens, const std::vector<TokenId>& prompt) {
    Engine engine(engine_options(artifact, capacity_tokens));
    const auto result = engine.generate(engine.prepare_tokens(prompt, false, false, true), greedy());
    require(result.generated_token_ids.size() == 8, "an arm stopped before its eight tokens");
    require(result.content_logprobs.size() >= 2, "an arm reported no log probabilities");
    Arm arm;
    arm.ids = result.generated_token_ids;
    arm.logprobs.reserve(result.content_logprobs.size());
    for (const auto& entry : result.content_logprobs) { arm.logprobs.push_back(entry.logprob); }
    arm.stats = engine.runtime_stats().streamed_kv;
    return arm;
}

bool same_geometry(const StreamedKvStats& a, const StreamedKvStats& b) {
    return a.resident_pages == b.resident_pages && a.ring_runs == b.ring_runs &&
           a.layer_resident == b.layer_resident;
}

int run() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "SKIP: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }

    std::string text;
    for (int i = 0; i < kSentenceRepeats; ++i) { text += kFiller; }
    text += "\nReply with exactly: ALPHA\n";
    std::vector<TokenId> prompt;
    {
        Engine tokenizer(engine_options(artifact, 262144));
        prompt = tokenizer.tokenize_text(text);
    }
    require(prompt.size() > 1024, "the prompt is too short to stream");

    // The same prompt and weights, two capacities: the narrower pool plans a smaller window and a
    // different ring depth, which is exactly the difference that used to change the answer.
    const Arm wide   = answer(artifact, 262144, prompt);
    const Arm narrow = answer(artifact, 196608, prompt);

    if (same_geometry(wide.stats, narrow.stats)) {
        std::cout << "SKIP: both capacities planned the same pool (resident "
                  << wide.stats.resident_pages << " pages/layer, " << wide.stats.ring_runs
                  << " ring runs), so the pair cannot compare two shapes\n";
        return 77;
    }
    require(wide.stats.enabled && narrow.stats.enabled,
            "an arm did not run the streamed space, so no pool shape was involved");

    require(wide.ids == narrow.ids, "the pool's shape changed the tokens the prompt produces");
    require(wide.logprobs.size() == narrow.logprobs.size(),
            "the arms reported a different number of log probabilities");
    require(std::memcmp(wide.logprobs.data(), narrow.logprobs.data(),
                        wide.logprobs.size() * sizeof(float)) == 0,
            "the pool's shape changed a log probability");

    std::fprintf(stderr,
                 "kv stream geometry: %u vs %u pages/layer, %u vs %u ring runs, %zu ids and %zu "
                 "logprobs identical\n",
                 wide.stats.resident_pages, narrow.stats.resident_pages, wide.stats.ring_runs,
                 narrow.stats.ring_runs, narrow.ids.size(), narrow.logprobs.size());
    return 0;
}

} // namespace

NINFER_GUARDED_TEST_MAIN(run)
