// A retained streamed context is a cache entry, not a lease.
//
// The first request of a served process publishes its context, and the next request may be a different
// conversation: base ninfer admits that request cold beside whatever is resident, and a streamed space
// must do the same instead of refusing to place it (measured: the second conversation of a served
// process failed admission, and with a second slot the stale handle took the process down with
// `KV activation reservation is stale`).
//
// The sequence here is the one that used to fail: a prompt, then a prompt sharing only its opening, then
// an unrelated one, then a longer one, then the first one again. Every request must be admitted, and the
// ones that cannot reuse the published context must produce what a cache-less run produces.
//
// Set NINFER_TEST_ARTIFACT to a Qwen3.8-27B v3 artifact. Without it the test skips with 77.

#include "guarded_main.h"

#include "ninfer/engine.h"
#include "ninfer/kv_cache_storage.h"

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
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
    options.prefill_a8      = true;
    options.invariant_attention = true;
    options.context_cache.enabled                           = retain;
    options.context_cache.mode                              = ninfer::ContextCacheMode::Legacy;
    options.context_cache.host_state_slots                  = retain ? 1U : 0U;
    options.context_cache.host_kv_capacity_bytes            = 0;
    options.context_cache.device_state_slots                = 0;
    options.context_cache.max_private_continuations         = retain ? 1U : 1U;
    options.context_cache.max_shared_prefixes               = 0;
    options.context_cache.max_long_anchors_per_continuation = 0;
    options.context_cache.host_cache_budget_bytes.reset();
    options.context_cache.disk_kv_path.clear();
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

std::vector<TokenId> prompt(Engine& tokenizer, const std::string& text) {
    return tokenizer.tokenize_text(text);
}

std::string text(const char* sentence, int repeats) {
    std::string out;
    for (int i = 0; i < repeats; ++i) { out += sentence; }
    return out;
}

int run() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "SKIP: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }

    std::vector<TokenId> first_prompt;
    std::vector<TokenId> partial;
    std::vector<TokenId> unrelated;
    std::vector<TokenId> longer;
    {
        Engine tokenizer(engine_options(artifact, false));
        first_prompt = prompt(tokenizer, text(
            "NInfer keeps every KV page of the context in pinned host memory. ", 60));
        unrelated = prompt(tokenizer, text("A completely different opening sentence here. ", 20));
        longer    = prompt(tokenizer, text("A completely different opening sentence here. ", 300));
        partial.assign(first_prompt.begin(), first_prompt.begin() + 400);
        const std::vector<TokenId> tail =
            prompt(tokenizer, text("Now answer a different question entirely. ", 30));
        partial.insert(partial.end(), tail.begin(), tail.end());
    }

    // What a run without any cache produces for the prompts that cannot reuse the published context.
    std::vector<TokenId> reference_unrelated;
    std::vector<TokenId> reference_longer;
    {
        Engine cold(engine_options(artifact, false));
        reference_unrelated =
            cold.generate(cold.prepare_tokens(unrelated), greedy(false)).generated_token_ids;
        reference_longer =
            cold.generate(cold.prepare_tokens(longer), greedy(false)).generated_token_ids;
    }

    Engine engine(engine_options(artifact, true));
    const auto first = engine.generate(engine.prepare_tokens(first_prompt), greedy(true));
    require(first.generated_token_ids.size() == 4, "the first request stopped early");
    const StreamedKvStats after_first = engine.runtime_stats().streamed_kv;
    require(after_first.retained, "the first request did not retain its context");

    struct Step {
        const char* name;
        const std::vector<TokenId>* prompt;
        const std::vector<TokenId>* expected;
    };
    const Step steps[] = {
        {"partial", &partial, nullptr},
        {"unrelated", &unrelated, &reference_unrelated},
        {"longer", &longer, &reference_longer},
        {"first again", &first_prompt, nullptr},
    };
    for (const Step& step : steps) {
        // Admission must succeed: a published context the head cannot reuse is a cache entry the next
        // request displaces, never a lease that refuses it.
        const auto result = engine.generate(engine.prepare_tokens(*step.prompt), greedy(true));
        require(result.generated_token_ids.size() == 4, "a later request stopped early");
        if (step.expected != nullptr) {
            require(result.generated_token_ids == *step.expected,
                    "a request admitted beside a retained context emitted different ids");
        }
        std::fprintf(stderr, "eviction: %s admitted, reused %u tokens\n", step.name,
                     result.reused_prompt_tokens);
    }
    std::fprintf(stderr, "eviction: four requests admitted beside one private continuation slot\n");
    return 0;
}

} // namespace

NINFER_GUARDED_TEST_MAIN(run)
