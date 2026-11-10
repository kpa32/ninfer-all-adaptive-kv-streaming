// C7 + retention: one process, several requests over a streamed context.
//
// The streamed space owns the device pool for the Engine's whole lifetime. Two shapes are checked:
//
//  1. Cold (`context_cache.enabled == false`, what `--no-prefix-reuse` asks for): the second request
//     starts after the first one's pages were released (KVStore::deactivate -> reset). The replay
//     must be bit-exact, and the second request must stream for itself (its rounds and bytes grow)
//     rather than read what the first left in the ring.
//  2. Retained (`context_cache` in the streamed shape: one Host StateImage, no host KV tier, no
//     device checkpoint slot): the finished request publishes a continuation whose KV is the pinned
//     host authority the space already owns, and the next prompt — the same prompt plus the tokens
//     the first request generated — continues that exact prefix. The second request must reuse the
//     whole prefix (prefill nothing) and produce the same tokens a cold run of that prompt produces.
//
// The prompt is 190 000 tokens because that is past the window a 262 144-capacity rk4v4 pool leaves
// (2905 pages = 185 920 tokens), so every long request slides the window and moves pages on the
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

EngineOptions engine_options(const char* artifact, bool retain) {
    EngineOptions options;
    options.artifact_path   = artifact;
    options.purpose         = EnginePurpose::Generation;
    options.max_context     = 262144;
    options.kv_capacity     = KvCapacityPolicy::automatic();
    options.kv_stream       = true;
    options.kv_cache        = KvCacheStorage::RotatedLloyd4KeyInt4Value;
    options.prefill_chunk   = 256;
    options.max_concurrency = 1;
    // The streamed retention shape: the pinned host authority is the KV tier, the endpoint's
    // StateImage is the only thing a checkpoint adds, and one continuation is retained.
    options.context_cache.enabled                           = retain;
    options.context_cache.mode                              = ninfer::ContextCacheMode::Legacy;
    options.context_cache.host_state_slots                  = retain ? 1U : 0U;
    options.context_cache.host_kv_capacity_bytes            = 0;
    options.context_cache.device_state_slots                = 0;
    options.context_cache.max_private_continuations         = 1;
    options.context_cache.max_shared_prefixes               = 0;
    options.context_cache.max_long_anchors_per_continuation = 0;
    options.context_cache.host_cache_budget_bytes.reset();
    options.context_cache.disk_kv_path.clear();
    return options;
}

// 1. Cold: the second request must replay bit-exactly and stream its own pages.
void cold_replay(const char* artifact) {
    Engine engine(engine_options(artifact, false));
    const auto first = engine.generate(engine.prepare_tokens(prompt()), greedy(8));
    const auto after_first = engine.runtime_stats().streamed_kv;
    require(first.prompt.prompt_tokens == 190000, "the first request did not take the whole prompt");
    require(first.generated_token_ids.size() == 8, "the first request stopped early");
    require(after_first.enabled, "the first request did not run the streamed space");
    require(after_first.rounds != 0 && after_first.upload_bytes != 0,
            "the first request streamed no round or moved no page");
    require(!after_first.retained, "a cold streamed request retained its context");

    // C9 runs on this request: with `NINFER_STREAM_VERIFY` set, every streamed page the ring receives
    // is copied back and compared with its host record (kv_stream_space.h, TransferStats). A record
    // that reports checks with no mismatch is the only proof that the block table, the copy and the
    // reuse gate all pointed the consumer at the bytes it was supposed to read - ids alone cannot
    // see a wrong page whose error is below the greedy margin.
#ifdef _WIN32
    _putenv_s("NINFER_STREAM_VERIFY", "1");
#else
    setenv("NINFER_STREAM_VERIFY", "1", 1);
#endif
    const auto second        = engine.generate(engine.prepare_tokens(prompt()), greedy(8));
    const auto after_second  = engine.runtime_stats().streamed_kv;
    require(second.generated_token_ids.size() == 8, "the second request stopped early");
    require(second.generated_token_ids == first.generated_token_ids,
            "the second request over the same streamed context emitted different ids");
    require(after_second.rounds > after_first.rounds,
            "the second request ran no streamed round of its own");
    require(after_second.upload_bytes > after_first.upload_bytes,
            "the second request moved no page of its own");
    require(after_second.verify_checks != 0, "the byte oracle compared no streamed page");
    require(after_second.verify_mismatches == 0,
            "the byte oracle found a streamed page that differs from its host record");

    std::cout << "kv stream cold: ids identical, rounds " << after_first.rounds << " -> "
              << after_second.rounds << ", upload bytes " << after_first.upload_bytes << " -> "
              << after_second.upload_bytes << ", window " << after_second.resident_pages
              << " pages/layer, byte oracle " << after_second.verify_checks << " checks / "
              << after_second.verify_mismatches << " mismatches\n";
}

// 2. Retained: the next request continues the published frontier and prefills none of it.
void retained_continuation(const char* artifact) {
    // The reference for the extended prompt, with nothing retained and reuse off.
    std::vector<TokenId> extended = prompt();
    {
        Engine engine(engine_options(artifact, false));
        const auto first = engine.generate(engine.prepare_tokens(extended), greedy(8));
        require(first.generated_token_ids.size() == 8, "the reference request stopped early");
        extended.insert(extended.end(), first.generated_token_ids.begin(),
                        first.generated_token_ids.end());
    }
    std::vector<TokenId> reference;
    {
        Engine engine(engine_options(artifact, false));
        reference = engine.generate(engine.prepare_tokens(extended), greedy(8)).generated_token_ids;
        require(reference.size() == 8, "the reference continuation stopped early");
    }

    Engine engine(engine_options(artifact, true));
    RequestOptions publish = greedy(8);
    publish.execution.allow_prefix_reuse = true;
    const auto first = engine.generate(engine.prepare_tokens(prompt()), publish);
    require(first.generated_token_ids.size() == 8, "the retained arm's first request stopped early");
    const auto after_first = engine.runtime_stats().streamed_kv;
    require(after_first.retained, "the streamed space did not retain the finished context");
    require(after_first.retained_frontier == 190008,
            "the retained frontier is not the finished request's execution frontier");

    RequestOptions reuse = greedy(8);
    reuse.execution.allow_prefix_reuse = true;
    const auto resumed = engine.generate(engine.prepare_tokens(extended), reuse);
    require(resumed.reused_prompt_tokens == after_first.retained_frontier,
            "the continuation did not reuse the whole retained prefix");
    require(resumed.generated_token_ids == reference,
            "the reused prefix changed the tokens the continuation produces");
    std::fprintf(stderr, "kv stream retained: reused %u of %zu prompt tokens, frontier %u\n",
                 resumed.reused_prompt_tokens, extended.size(), after_first.retained_frontier);
}

int run() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "SKIP: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    cold_replay(artifact);
    retained_continuation(artifact);
    return 0;
}

} // namespace

NINFER_GUARDED_TEST_MAIN(run)
