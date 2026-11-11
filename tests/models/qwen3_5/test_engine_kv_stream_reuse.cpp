// The reuse cases a streamed run meets in ordinary use, each checked against a cache-less run.
//
// The eviction test (test_engine_kv_stream_eviction.cpp) covers admission: a published context the
// head cannot reuse must never refuse the next request. This file covers what the *bytes* may be
// used for once a request is admitted, one case per situation an operator actually produces:
//
//   repeating a prompt        - the turn-closure path must reuse, and the ids stay the same
//   cancelling a prefill      - the next conversation is admitted and unaffected
//   cancelling then repeating - a rewrite of the same prompt, which is where a claim on another
//                               conversation's pages would show up as a silent wrong answer
//   two conversations         - A, B, then A again: whatever gets reused, the ids match a cold run
//   a shorter request first   - the space rewinds below a published frontier before the long one
//
// Every case compares against a *cold* engine run on the same artifact, so a stale claim cannot pass
// by producing plausible text: it has to match token for token.
//
// Set NINFER_TEST_ARTIFACT to a Qwen3.8-27B v3 artifact. Without it the test skips with 77.

#include "guarded_main.h"

#include "ninfer/engine.h"
#include "ninfer/kv_cache_storage.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <thread>
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

EngineOptions engine_options(const char* artifact, bool retain) {
    EngineOptions options;
    options.artifact_path       = artifact;
    options.purpose             = EnginePurpose::Generation;
    options.max_context         = 262144;
    options.kv_capacity         = KvCapacityPolicy::automatic();
    options.kv_stream           = true;
    options.kv_cache            = KvCacheStorage::RotatedLloyd4KeyInt4Value;
    options.prefill_chunk       = 256;
    options.max_concurrency     = 1;
    options.prefill_a8          = true;
    options.invariant_attention = true;
    options.context_cache.enabled                           = retain;
    options.context_cache.mode                              = ninfer::ContextCacheMode::Legacy;
    options.context_cache.host_state_slots                  = retain ? 4U : 0U;
    options.context_cache.host_kv_capacity_bytes            = 0;
    options.context_cache.device_state_slots                = 0;
    options.context_cache.max_private_continuations         = retain ? 2U : 1U;
    options.context_cache.max_shared_prefixes               = 0;
    options.context_cache.max_long_anchors_per_continuation = retain ? 2U : 0U;
    options.context_cache.host_cache_budget_bytes.reset();
    options.context_cache.disk_kv_path.clear();
    // Progress anchors are the restore points a long single prompt leaves inside itself; without
    // them a cancelled prefill has nothing to continue from (kv-stream-context-cache.md 19.2).
    options.context_cache.progress_anchor_tokens = retain ? 16384U : 0U;
    return options;
}

RequestOptions greedy(bool reuse) {
    RequestOptions out;
    out.execution.requested_output_tokens = 4;
    out.execution.sampling.temperature     = 0;
    out.execution.allow_prefix_reuse       = reuse;
    out.stop.include_model_defaults        = false;
    return out;
}

std::string text(const char* sentence, int repeats) {
    std::string out;
    for (int i = 0; i < repeats; ++i) { out += sentence; }
    return out;
}

struct ColdArms {
    std::vector<TokenId> repeated;    // the prompt sent twice
    std::vector<TokenId> unrelated;   // the second conversation
    std::vector<TokenId> long_report; // the prompt cancelled mid-prefill
};

int run() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "SKIP: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }

    std::vector<TokenId> report;
    std::vector<TokenId> other;
    std::vector<TokenId> long_report;
    {
        Engine tokenizer(engine_options(artifact, false));
        report        = tokenizer.tokenize_text(text(
            "NInfer keeps every KV page of the context in pinned host memory. ", 60));
        other         = tokenizer.tokenize_text(text("A completely different opening sentence. ", 40));
        long_report   = tokenizer.tokenize_text(text(
            "The quarterly report lists every shipped subsystem and its measured latency. ", 3000));
    }

    // The same three prompts without any cache: the ids a streamed run must also produce.
    ColdArms cold_ids;
    {
        Engine cold(engine_options(artifact, false));
        cold_ids.repeated =
            cold.generate(cold.prepare_tokens(report), greedy(false)).generated_token_ids;
        cold_ids.unrelated =
            cold.generate(cold.prepare_tokens(other), greedy(false)).generated_token_ids;
        cold_ids.long_report =
            cold.generate(cold.prepare_tokens(long_report), greedy(false)).generated_token_ids;
    }

    Engine engine(engine_options(artifact, true));

    // 0. Two conversations on a fresh catalogue, then the first one again. This runs first because
    // the cases below publish and consume entries, and a catalogue of two slots cannot be expected
    // to still hold the first conversation after five of them: this is the only arrangement in which
    // 'return to A' has A's own entry to find. A asks for its end anchor, because a raw prompt has no
    // message structure and an endpoint past the prompt cannot serve an identical one.
    {
        const auto a1 = engine.generate(
            engine.prepare_tokens(report, true, /*anchor_prompt_end=*/true), greedy(true));
        require(a1.generated_token_ids == cold_ids.repeated, "the first A differs from a cold run");
        const auto b1 = engine.generate(engine.prepare_tokens(other), greedy(true));
        require(b1.generated_token_ids == cold_ids.unrelated, "B differs from a cold run");
        const auto a2 = engine.generate(
            engine.prepare_tokens(report, true, /*anchor_prompt_end=*/true), greedy(true));
        require(a2.generated_token_ids == cold_ids.repeated,
                "the second A differs from a cold run");
        std::fprintf(stderr, "reuse: A -> B -> A on a fresh catalogue reused %u tokens\n",
                     a2.reused_prompt_tokens);
    }

    // 1. Cancelling past the progress-anchor grid and then sending the same prompt again: the
    // cancelled lane's own checkpoint is the only thing that can serve this request, so it runs
    // first, while the one private continuation slot is still empty. A second engine would need
    // the weights a second time on the same device.
    {
        auto handle = engine.submit(engine.prepare_tokens(long_report), greedy(true));
        std::this_thread::sleep_for(std::chrono::milliseconds(10000));
        handle = {};
        const auto rewrite = engine.generate(engine.prepare_tokens(long_report, true, /*anchor_prompt_end=*/true), greedy(true));
        require(rewrite.generated_token_ids == cold_ids.long_report,
                "the request after a cancel emitted different ids");
        std::fprintf(stderr, "reuse: prompt after cancel reused %u tokens\n",
                     rewrite.reused_prompt_tokens);
    }

    // 2. Repeating a prompt: the raw-token prompt has no message structure, so the engine anchors
    // its end only when asked (prepare_tokens' anchor_prompt_end). Whatever it reuses must not
    // change the answer.
    auto anchored_report = engine.prepare_tokens(report, true, /*anchor_prompt_end=*/true);
    const auto first = engine.generate(std::move(anchored_report), greedy(true));
    require(first.generated_token_ids == cold_ids.repeated, "the first answer differs from a cold run");
    auto anchored_again = engine.prepare_tokens(report, true, /*anchor_prompt_end=*/true);
    const auto again = engine.generate(std::move(anchored_again), greedy(true));
    require(again.generated_token_ids == cold_ids.repeated, "a repeated prompt emitted different ids");
    std::fprintf(stderr, "reuse: repeated prompt reused %u tokens\n", again.reused_prompt_tokens);

    // 3. Cancelling a prefill: the cancelled work must not poison the next conversation.
    {
        auto handle = engine.submit(engine.prepare_tokens(long_report), greedy(true));
        std::this_thread::sleep_for(std::chrono::milliseconds(1500));
        handle = {};  // destroying an unconsumed handle cancels its request
    }
    const auto beside = engine.generate(engine.prepare_tokens(other), greedy(true));
    require(beside.generated_token_ids == cold_ids.unrelated,
            "a conversation beside a cancelled one emitted different ids");
    std::fprintf(stderr, "reuse: unrelated after cancel reused %u tokens\n",
                 beside.reused_prompt_tokens);


    // 4. Two conversations, then the first one again: whichever context the space kept, the answer
    // must match a cold run, and the request must be admitted.
    const auto second_other = engine.generate(engine.prepare_tokens(other), greedy(true));
    require(second_other.generated_token_ids == cold_ids.unrelated,
            "the second visit of a conversation emitted different ids");
    const auto back_to_report = engine.generate(engine.prepare_tokens(report), greedy(true));
    require(back_to_report.generated_token_ids == cold_ids.repeated,
            "returning to the first conversation emitted different ids");
    std::fprintf(stderr, "reuse: A -> B -> A reused %u tokens\n", back_to_report.reused_prompt_tokens);

    // 5. A request shorter than the space's frontier, then a longer one: the rewind must leave the
    // longer request admitted rather than failing on a frontier the space no longer serves.
    const auto shorter = engine.generate(engine.prepare_tokens(other), greedy(true));
    require(shorter.generated_token_ids == cold_ids.unrelated,
            "a shorter request before a longer one emitted different ids");
    const auto longer = engine.generate(engine.prepare_tokens(long_report), greedy(true));
    require(longer.generated_token_ids == cold_ids.long_report,
            "a longer request after a rewind emitted different ids");

    const auto stats = engine.runtime_stats().streamed_kv;
    std::fprintf(stderr,
                 "reuse: five cases admitted; upload %llu B, download %llu B, verify %llu/%llu\n",
                 static_cast<unsigned long long>(stats.upload_bytes),
                 static_cast<unsigned long long>(stats.download_bytes),
                 static_cast<unsigned long long>(stats.verify_checks),
                 static_cast<unsigned long long>(stats.verify_mismatches));
    require(stats.verify_mismatches == 0, "the byte oracle reported a mismatch");
    return 0;
}

} // namespace

NINFER_GUARDED_TEST_MAIN(run)
