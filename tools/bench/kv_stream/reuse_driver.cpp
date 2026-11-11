// Two-request streamed retention check: exact-prefix continuation through the public Engine.
#include "ninfer/engine.h"
#include "ninfer/kv_cache_storage.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;

static void require(bool ok, const char* what) {
    if (!ok) { throw std::runtime_error(what); }
}

static EngineOptions options(const char* artifact, bool retain) {
    EngineOptions o;
    o.artifact_path   = artifact;
    o.purpose         = EnginePurpose::Generation;
    o.max_context     = static_cast<std::uint32_t>(std::getenv("DRIVER_MAX_CONTEXT")
                                                          ? std::atoi(std::getenv("DRIVER_MAX_CONTEXT"))
                                                          : 262144);
    o.kv_capacity     = KvCapacityPolicy::automatic();
    o.kv_stream       = true;
    o.kv_cache        = KvCacheStorage::RotatedInt8KeyInt4ValueGroup64;
    o.prefill_chunk   = 256;
    o.gdn_state_fp16  = true;
    o.invariant_attention = true;
    o.embedding_residency = EmbeddingResidency::Host;
    o.max_concurrency = 1;
    o.context_cache.enabled                           = retain;
    o.context_cache.mode                              = ContextCacheMode::Legacy;
    o.context_cache.host_state_slots                  = retain ? 1U : 0U;
    o.context_cache.host_kv_capacity_bytes            = 0;
    // A retained endpoint lives on the host shelf: one host StateImage, no device state slot
    // (kv_stream_guards.h rejects the device slot beside a streamed window).
    o.context_cache.device_state_slots                = 0;
    o.context_cache.max_private_continuations         = retain ? 1U : 1U;
    o.context_cache.max_shared_prefixes               = 0;
    o.context_cache.max_long_anchors_per_continuation = 0;
    o.context_cache.max_cache_markers_per_request     = 0;
    o.context_cache.host_cache_budget_bytes.reset();
    o.context_cache.disk_kv_path.clear();
    o.speculative.backend     = SpeculativeBackend::Mtp;
    o.speculative.draft_tokens = 3;
    o.speculative.mtp_policy   = MtpDraftPolicy::Adaptive;
    o.speculative.ngram_draft_tokens = 15;
    o.speculative.proposal_head      = ProposalHead::Optimized;
    o.prefill_a8                     = true;
    return o;
}

static RequestOptions greedy(std::uint32_t n, bool reuse) {
    RequestOptions r;
    r.execution.requested_output_tokens = n;
    r.execution.sampling.temperature     = 0;
    r.execution.allow_prefix_reuse       = reuse;
    r.execution.speculative_drafting     = true;
    r.stop.include_model_defaults        = false;
    return r;
}

// The served warmup: one token, no draft, no cache participation.
static RequestOptions warmup_options() {
    RequestOptions r = greedy(1, false);
    r.execution.speculative_drafting = false;
    return r;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s ARTIFACT\n", argv[0]); return 2; }
    const char* artifact = argv[1];
    const std::vector<TokenId> prompt = [&] {
        // The served prompt: the same text the HTTP arm sends, tokenized by the artifact.
        static constexpr const char* kFiller =
            "NInfer keeps every KV page of the context in pinned host memory and streams "
            "the newest window onto the device. ";
        std::string text;
        for (int i = 0; i < 200; ++i) { text += kFiller; }
        text += "\nReply with exactly: ALPHA\n";
        Engine tokenizer(options(artifact, false));
        return tokenizer.tokenize_text(text);
    }();
    std::fprintf(stderr, "prompt_tokens=%zu\n", prompt.size());
    std::vector<TokenId> extended      = prompt;
    std::vector<TokenId> reference;
    std::vector<TokenId> cold_first;
    SpeculativeStats cold_spec;
    {
        Engine cold(options(artifact, false));
        // The served process warms up first: one token, no draft, no cache participation.
        cold.generate(cold.prepare_tokens({248045, 846, 198}, false, false, false), warmup_options());
        const auto first = cold.generate(
            cold.prepare_tokens(prompt, /*allow_prefix_identity=*/false,
                                /*anchor_prompt_end=*/false, /*speculative_drafting=*/true),
            greedy(8, false));
        require(first.generated_token_ids.size() == 8, "cold first request stopped early");
        cold_first = first.generated_token_ids;

        extended.insert(extended.end(), first.generated_token_ids.begin(),
                        first.generated_token_ids.end());
        reference =
            cold.generate(cold.prepare_tokens(extended), greedy(8, false)).generated_token_ids;
        require(reference.size() == 8, "cold reference stopped early");
    }
    Engine engine(options(artifact, true));
    // The same warmup, on the engine that participates in the cache.
    engine.generate(engine.prepare_tokens({248045, 846, 198}, false, false, false),
                    warmup_options());
    const auto first = engine.generate(
        engine.prepare_tokens(prompt, /*allow_prefix_identity=*/true,
                              /*anchor_prompt_end=*/true, /*speculative_drafting=*/true),
        greedy(8, true));
    require(first.generated_token_ids.size() == 8, "retained first request stopped early");
    bool same_first = cold_first.size() == first.generated_token_ids.size();
    if (same_first) {
        for (std::size_t i = 0; i < cold_first.size(); ++i) {
            if (cold_first[i] != first.generated_token_ids[i]) {
                std::fprintf(stderr, "first divergence at %zu: cold=%d retained=%d\n", i,
                             static_cast<int>(cold_first[i]),
                             static_cast<int>(first.generated_token_ids[i]));
                same_first = false;
                break;
            }
        }
    }
    std::fprintf(stderr, "first_equal=%d cold=%zu retained=%zu\n", same_first ? 1 : 0,
                 cold_first.size(), first.generated_token_ids.size());
    const StreamedKvStats after_first = engine.runtime_stats().streamed_kv;
    std::fprintf(stderr, "cold ids:");
    for (const TokenId t : cold_first) { std::fprintf(stderr, " %d", static_cast<int>(t)); }
    std::fprintf(stderr, "\nretained ids:");
    for (const TokenId t : first.generated_token_ids) {
        std::fprintf(stderr, " %d", static_cast<int>(t));
    }
    std::fprintf(stderr, "\n");
    std::fprintf(stderr, "retained=%d frontier=%u window=%u\n", after_first.retained ? 1 : 0,
                 after_first.retained_frontier, after_first.resident_pages);
    require(after_first.retained, "the streamed space did not retain the finished context");
    require(after_first.retained_frontier != 0, "the retained frontier is zero");
    const auto resumed = engine.generate(engine.prepare_tokens(extended), greedy(8, true));
    std::fprintf(stderr, "reused=%u frontier=%u equal_ids=%d\n", resumed.reused_prompt_tokens,
                 after_first.retained_frontier,
                 resumed.generated_token_ids == reference ? 1 : 0);
    require(resumed.reused_prompt_tokens >= after_first.retained_frontier - 1,
            "the continuation reused less than the retained frontier");
    require(resumed.generated_token_ids == reference,
            "the reused prefix changed the continuation's tokens");
    std::fprintf(stderr, "OK\n");
    return 0;
}
