// C7: one process, two requests over a streamed context.
//
// The streamed space owns the device pool for the Engine's whole lifetime, so a second request
// starts after the first one's pages were released (KVStore::deactivate -> reset). This test
// checks that the replay is bit-exact: the same prompt must produce the same greedy ids, and the
// second request must stream for itself (its rounds and bytes grow) rather than read what the
// first left in the ring.
//
// The prompt is 190 000 tokens because that is past the window a 262 144-capacity rk4v4 pool
// leaves (2905 pages = 185 920 tokens), so both requests slide the window and move pages on the
// delivered path (one whole-layer copy per round).
//
// Set NINFER_TEST_ARTIFACT to a Qwen3.8-27B v3 artifact. Without it the test skips with 77.

#include "guarded_main.h"

#include "ninfer/engine.h"
#include "ninfer/kv_cache_storage.h"

#include <cstdint>
#include <cstdlib>
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
using ninfer::TokenId;

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

std::vector<TokenId> prompt() {
    std::vector<TokenId> out{248045, 846, 198, 5834, 248046, 198};
    const std::vector<TokenId> sentence{791, 5296, 315, 279, 1917, 374, 682, 198};
    out.reserve(192000);
    while (out.size() < 190000) { out.insert(out.end(), sentence.begin(), sentence.end()); }
    out.resize(190000);
    return out;
}

RequestOptions greedy(std::uint32_t outputs) {
    RequestOptions out;
    out.execution.requested_output_tokens = outputs;
    out.execution.sampling.temperature   = 0;
    out.execution.allow_prefix_reuse     = false;
    out.stop.include_model_defaults      = false;
    return out;
}

int run() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "SKIP: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }

    EngineOptions options;
    options.artifact_path  = artifact;
    options.purpose        = EnginePurpose::Generation;
    options.max_context    = 262144;
    options.kv_capacity    = KvCapacityPolicy::automatic();
    options.kv_stream      = true;
    options.kv_cache       = KvCacheStorage::RotatedLloyd4KeyInt4Value;
    options.prefill_chunk  = 256;
    options.max_concurrency = 1;
    options.context_cache.enabled = false;
    Engine engine(options);

    const auto first = engine.generate(engine.prepare_tokens(prompt()), greedy(8));
    const auto after_first = engine.runtime_stats().streamed_kv;
    require(first.prompt.prompt_tokens == 190000, "the first request did not take the whole prompt");
    require(first.generated_token_ids.size() == 8, "the first request stopped early");
    require(after_first.enabled, "the first request did not run the streamed space");
    require(after_first.rounds != 0 && after_first.upload_bytes != 0,
            "the first request streamed no round or moved no page");

    // The same prompt again, after the first context was deactivated and the space reset. If any
    // page of the first request survived, these bytes would not be the same answer.
    const auto second = engine.generate(engine.prepare_tokens(prompt()), greedy(8));
    const auto after_second = engine.runtime_stats().streamed_kv;
    require(second.generated_token_ids.size() == 8, "the second request stopped early");
    require(second.generated_token_ids == first.generated_token_ids,
            "the second request over the same streamed context emitted different ids");
    require(after_second.rounds > after_first.rounds,
            "the second request ran no streamed round of its own");
    require(after_second.upload_bytes > after_first.upload_bytes,
            "the second request moved no page of its own");

    std::cout << "kv stream serial: ids identical, rounds " << after_first.rounds << " -> "
              << after_second.rounds << ", upload bytes " << after_first.upload_bytes << " -> "
              << after_second.upload_bytes << ", window " << after_second.resident_pages
              << " pages/layer, shape " << after_second.shape << '\n';
    return 0;
}

} // namespace

NINFER_GUARDED_TEST_MAIN(run)
