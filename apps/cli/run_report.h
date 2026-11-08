#pragma once

// The machine-readable record of one CLI run: what it asked for, what the card is, what the
// request moved and waited for, and the per-round series the streamed KV loop produced.
//
// The names here are also the human summary's names; both read one definition so a value cannot be
// spelled two ways in two places.

#include "options.h"

#include "ninfer/engine.h"
#include "ninfer/observability/resources.h"
#include "ninfer/observability/snapshot.h"

#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

namespace ninfer::cli {

std::string finish_reason_name(FinishReason reason);
std::string kv_cache_name(KvCacheStorage storage);
std::string kv_capacity_mode_name(KvCapacityMode mode);

// Everything the report needs, handed over as the values the CLI already holds.
struct RunReport {
    const Options& options;
    const GenerationResult& result;
    const ResolvedSamplingParameters& sampling;
    const MemorySummary& memory;
    const RuntimeStats& stats;
    const LoadSummary& load;
    std::vector<std::string> argv;
    std::chrono::system_clock::time_point started;
    std::chrono::system_clock::time_point finished;
};

// Writes the report as one JSON object with two-space indentation. Throws std::runtime_error when
// the path cannot be written.
void write_run_report(const std::filesystem::path& path, const RunReport& report);

} // namespace ninfer::cli
