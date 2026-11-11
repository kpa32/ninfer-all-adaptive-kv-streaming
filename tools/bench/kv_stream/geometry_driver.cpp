// Controlled-geometry check: two streamed Engines of this build answer the same prompt, and the only
// difference between them is how much device memory was reserved before the engine was created, so
// their pools (window depth, ring depth) differ. The answer must not: a page is named by its layer's
// base plus its position in the window, so a layout that changed a name without moving the bytes
// answered a different question than the prompt asked
// (bench/results/kv-stream/layout-rename-defect.md). Greedy generation with logprobs on reports the
// model's own log p per token, so a difference here is a difference in the numerics, not in sampling.
//
//   geometry_driver ARTIFACT
//
// LP_RESERVE1 / LP_RESERVE2  MiB reserved before each engine (0 keeps them equal)
// LP_ARM2_RETAINED=1         the second arm keeps its context (context cache + prefix reuse)
// LP_PROMPT_TOKENS / LP_CHUNK / LP_MAX_CONTEXT  prompt recipe and pool size
// LP_NO_A8 / LP_NO_SPEC / LP_HOST_SLOTS         single-knob controls
//

#include "ninfer/engine.h"
#include "ninfer/kv_cache_storage.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace ninfer;

static std::string filler_prompt() {
    static constexpr const char* kFiller =
        "NInfer keeps every KV page of the context in pinned host memory and streams "
        "the newest window onto the device. ";
    std::string text;
    for (int i = 0; i < 200; ++i) { text += kFiller; }
    text += "\nReply with exactly: ALPHA\n";
    return text;
}

static EngineOptions options(const char* artifact, bool retain) {
    EngineOptions o;
    o.artifact_path       = artifact;
    o.purpose             = EnginePurpose::Generation;
    o.max_context = static_cast<std::uint32_t>(std::getenv("LP_MAX_CONTEXT") ? std::atoi(std::getenv("LP_MAX_CONTEXT")) : 262144);
    o.kv_capacity         = KvCapacityPolicy::automatic();
    o.kv_stream           = true;
    o.kv_cache            = KvCacheStorage::RotatedInt8KeyInt4ValueGroup64;
    o.prefill_chunk = static_cast<std::uint32_t>(std::getenv("LP_CHUNK") ? std::atoi(std::getenv("LP_CHUNK")) : 256);
    o.max_concurrency     = 1;
    o.gdn_state_fp16      = true;
    o.invariant_attention = true;
    o.prefill_a8 = std::getenv("LP_NO_A8") == nullptr;
    o.embedding_residency = EmbeddingResidency::Host;
    o.context_cache.enabled                           = retain;
    o.context_cache.mode                              = ContextCacheMode::Legacy;
    o.context_cache.host_state_slots = retain ? static_cast<std::uint32_t>(
                                                    std::getenv("LP_HOST_SLOTS")
                                                        ? std::atoi(std::getenv("LP_HOST_SLOTS"))
                                                        : 1)
                                                : 0U;
    o.context_cache.host_kv_capacity_bytes            = 0;
    o.context_cache.device_state_slots                = 0;
    o.context_cache.max_private_continuations         = 1;
    o.context_cache.max_shared_prefixes               = 0;
    o.context_cache.max_long_anchors_per_continuation = 0;
    o.context_cache.max_cache_markers_per_request     = 0;
    o.context_cache.host_cache_budget_bytes.reset();
    o.context_cache.disk_kv_path.clear();
    if (std::getenv("LP_NOANCHORS") != nullptr) {
        o.context_cache.endpoint_anchors              = false;
        o.context_cache.progress_anchor_tokens        = 0;
        o.context_cache.automatic_long_anchors        = false;
        o.context_cache.branch_anchors                = false;
    }
    o.speculative.backend = std::getenv("LP_NO_SPEC") ? SpeculativeBackend::None
                                                       : SpeculativeBackend::Mtp;
    o.speculative.draft_tokens = std::getenv("LP_NO_SPEC") ? 0U : 3U;
    o.speculative.mtp_policy         = MtpDraftPolicy::Adaptive;
    o.speculative.ngram_draft_tokens = 15;
    o.speculative.proposal_head      = ProposalHead::Optimized;
    return o;
}

static RequestOptions greedy(std::uint32_t n, bool reuse) {
    RequestOptions r;
    r.execution.requested_output_tokens = n;
    r.execution.sampling.temperature     = 0;
    r.execution.allow_prefix_reuse       = reuse;
    r.execution.speculative_drafting     = true;
    r.execution.logprobs                 = true;
    r.stop.include_model_defaults        = false;
    return r;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s ARTIFACT\n", argv[0]); return 2; }
    const char* artifact = argv[1];
    std::vector<TokenId> tokens;
    {
        Engine tokenizer(options(artifact, false));
        tokens = tokenizer.tokenize_text(filler_prompt());
        if (const char* want = std::getenv("LP_PROMPT_TOKENS")) {
            tokens.resize(static_cast<std::size_t>(std::atoi(want)));
        }
    }
    std::printf("prompt_tokens=%zu\n", tokens.size());

    StreamedKvStats cold_geom;
    StreamedKvStats ret_geom;
    GenerationResult cold;
    if (std::getenv("LP_SKIP_COLD") == nullptr) {
        Engine engine(options(artifact, false));
        cold = engine.generate(engine.prepare_tokens(tokens, false, false, true), greedy(8, false));
        cold_geom = engine.runtime_stats().streamed_kv;
    }
    GenerationResult retained; // also the second cache-less arm under LP_COLD_TWICE
    if (std::getenv("LP_SKIP_RETAINED") == nullptr) {
        const bool twice = std::getenv("LP_COLD_TWICE") != nullptr;
        std::fprintf(stderr, "=== ARM 2 START ===\n");
        Engine engine(options(artifact, twice ? false : true));
        const bool identity = twice ? false : std::getenv("LP_IDENTITY_OFF") == nullptr;
        retained = engine.generate(engine.prepare_tokens(tokens, identity, false, true),
                                   greedy(8, identity));
        ret_geom = engine.runtime_stats().streamed_kv;
    }
    std::printf("cold     : ids=%zu lp=%zu finish=%d\n", cold.generated_token_ids.size(),
                cold.content_logprobs.size(), static_cast<int>(cold.finish_reason));
    std::printf("retained : ids=%zu lp=%zu finish=%d\n", retained.generated_token_ids.size(),
                retained.content_logprobs.size(), static_cast<int>(retained.finish_reason));
    // Controlled geometry pair: reserve device memory before each engine so the automatic pool
    // lands on a different window, then compare the two answers.
    const auto reserve_mb = [](long mib) -> void* {
        if (mib <= 0) { return nullptr; }
        void* ptr = nullptr;
        const cudaError_t err = cudaMalloc(&ptr, static_cast<std::size_t>(mib) << 20);
        std::printf("reserve %ld MiB: %s\n", mib, err == cudaSuccess ? "ok" : "failed");
        return ptr;
    };
    const long first_mib  = std::getenv("LP_RESERVE1") ? std::atol(std::getenv("LP_RESERVE1")) : 0;
    const long second_mib = std::getenv("LP_RESERVE2") ? std::atol(std::getenv("LP_RESERVE2")) : 0;
    GenerationResult a, b;
    StreamedKvStats ga, gb;
    {
        void* hold = reserve_mb(first_mib);
        Engine engine(options(artifact, false));
        a  = engine.generate(engine.prepare_tokens(tokens, false, false, true), greedy(8, false));
        ga = engine.runtime_stats().streamed_kv;
        if (hold != nullptr) { (void)cudaFree(hold); }
        std::printf("arm1: reserve=%ld resident=%u ring=%u id1=%d logp1=%.9g\n", first_mib,
                    ga.resident_pages, ga.ring_runs,
                    a.content_logprobs.size() > 1 ? static_cast<int>(a.content_logprobs[1].id) : -1,
                    a.content_logprobs.size() > 1 ? double(a.content_logprobs[1].logprob) : 0.0);
    }
    {
        void* hold = reserve_mb(second_mib);
        const bool retained_arm = std::getenv("LP_ARM2_RETAINED") != nullptr;
        Engine engine(options(artifact, retained_arm));
        b = engine.generate(
            engine.prepare_tokens(tokens, retained_arm, false, true), greedy(8, retained_arm));
        gb = engine.runtime_stats().streamed_kv;
        if (hold != nullptr) { (void)cudaFree(hold); }
        std::printf("arm2: reserve=%ld resident=%u ring=%u id1=%d logp1=%.9g\n", second_mib,
                    gb.resident_pages, gb.ring_runs,
                    b.content_logprobs.size() > 1 ? static_cast<int>(b.content_logprobs[1].id) : -1,
                    b.content_logprobs.size() > 1 ? double(b.content_logprobs[1].logprob) : 0.0);
    }
    const bool equal = a.generated_token_ids == b.generated_token_ids;
    bool same_bits   = equal && a.content_logprobs.size() == b.content_logprobs.size();
    for (std::size_t i = 0; same_bits && i < a.content_logprobs.size(); ++i) {
        same_bits = std::memcmp(&a.content_logprobs[i].logprob, &b.content_logprobs[i].logprob,
                                sizeof(float)) == 0;
    }
    std::printf("ids equal=%d logprobs equal=%d\n", equal ? 1 : 0, same_bits ? 1 : 0);
    return same_bits ? 0 : 1;
}
