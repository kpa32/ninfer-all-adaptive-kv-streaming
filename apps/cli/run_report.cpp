#include "run_report.h"

#include "ninfer_build_id.h"

#include "ninfer/observability/execution.h"
#include "product/speculative_options.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <stdexcept>
#include <string>

namespace ninfer::cli {
namespace {

using json = nlohmann::json;

std::string utc(std::chrono::system_clock::time_point point) {
    const std::time_t seconds = std::chrono::system_clock::to_time_t(point);
    std::tm broken{};
#ifdef _WIN32
    gmtime_s(&broken, &seconds);
#else
    gmtime_r(&seconds, &broken);
#endif
    char buffer[32];
    if (std::strftime(buffer, sizeof buffer, "%Y-%m-%dT%H:%M:%SZ", &broken) == 0) {
        return std::string{};
    }
    return buffer;
}

double seconds_between(std::chrono::system_clock::time_point from,
                       std::chrono::system_clock::time_point to) {
    return std::chrono::duration<double>(to - from).count();
}

json arena(const ArenaMemorySummary& value) {
    return json{{"capacity_bytes", value.capacity_bytes},
                {"used_bytes", value.used_bytes},
                {"peak_used_bytes", value.peak_used_bytes}};
}

json memory_summary(const MemorySummary& memory) {
    json out;
    out["max_context"]                 = memory.max_context;
    out["kv_capacity_mode"]            = kv_capacity_mode_name(memory.kv_capacity_mode);
    out["kv_capacity"]                 = memory.kv_capacity;
    out["kv_capacity_page_groups"]     = memory.kv_capacity_page_groups;
    out["kv_capacity_max_page_groups"] = memory.kv_capacity_max_page_groups;
    out["kv_cache"]                    = kv_cache_name(memory.kv_cache);
    out["kv_payload_bytes"]            = memory.kv_payload_bytes;
    out["weights"]                     = arena(memory.weights);
    out["sequence"]                    = arena(memory.sequence);
    out["workspace"]                   = arena(memory.workspace);
    out["workspace_logical_peak_bytes"]     = memory.workspace_logical_peak_bytes;
    out["minimum_runtime_reservation_bytes"] = memory.minimum_runtime_reservation_bytes;
    out["runtime_reservation_bytes"]        = memory.runtime_reservation_bytes;
    out["kv_capacity_increment_bytes"]      = memory.kv_capacity_increment_bytes;
    out["available_after_weights_bytes"]    = memory.available_after_weights_bytes;
    out["available_after_startup_bytes"]    = memory.available_after_startup_bytes;
    out["kv_capacity_headroom_bytes"]       = memory.kv_capacity_headroom_bytes;
    out["planned_slack_bytes"]              = memory.planned_slack_bytes;
    out["cuda_graph_allowance_bytes"]       = memory.cuda_graph_allowance_bytes;
    if (memory.cuda_graph_measured_bytes) {
        out["cuda_graph_measured_bytes"] = *memory.cuda_graph_measured_bytes;
    }
    out["host_kv_capacity_bytes"] = memory.host_kv_capacity_bytes;
    out["host_kv_occupied_bytes"] = memory.host_kv_occupied_bytes;
    out["host_kv_page_group_bytes"] = memory.host_kv_page_group_bytes;
    out["host_state_capacity_slots"] = memory.host_state_capacity_slots;
    out["host_state_occupied_slots"] = memory.host_state_occupied_slots;
    out["host_state_image_bytes"]    = memory.host_state_image_bytes;
    out["host_cache_budget_bytes"]   = memory.host_cache_budget_bytes;
    out["expert_cache_bytes"]        = memory.expert_cache_bytes;
    out["devices"] = json::array();
    for (const DeviceMemorySummary& device : memory.devices) {
        out["devices"].push_back(json{{"device", device.device},
                                      {"weights", arena(device.weights)},
                                      {"sequence", arena(device.sequence)},
                                      {"workspace", arena(device.workspace)},
                                      {"expert_cache_bytes", device.expert_cache_bytes}});
    }
    return out;
}

json stream_stats(const StreamedKvStats& stream) {
    if (!stream.enabled) { return json{{"enabled", false}}; }
    json out;
    out["enabled"]           = true;
    out["resident_pages"]    = stream.resident_pages;
    out["layer_resident"]    = stream.layer_resident;
    // The live frontier is reset when the request ends, so an absent value says "not recorded", not
    // "zero pages": printing 0 would read as an empty context beside a full round series.
    if (stream.active_pages != 0) { out["active_pages"] = stream.active_pages; }
    if (stream.shape != 0) { out["shape"] = kv_stream_shape_label(stream.shape); }
    out["layout_generation"] = stream.layout_generation;
    // A finished request's context still lives in the space; the next request either continues from
    // this frontier or starts from an empty one.
    out["retained"]          = stream.retained;
    if (stream.retained) { out["retained_frontier"] = stream.retained_frontier; }
    out["streamed_pages"]    = stream.streamed_pages;
    out["layers"]            = stream.layers;
    out["ring_runs"]         = stream.ring_runs;
    out["run_pages"]         = stream.run_pages;
    out["pages_per_layer"]   = stream.pages_per_layer;
    out["page_bytes"]        = stream.page_bytes;
    out["host_bytes"]        = stream.host_bytes;
    out["device_bytes"]      = stream.device_bytes;
    out["window_bytes"]      = stream.window_bytes;
    out["ring_bytes"]        = stream.ring_bytes;
    out["rounds"]            = stream.rounds;
    out["upload_bytes"]      = stream.upload_bytes;
    out["download_bytes"]    = stream.download_bytes;
    out["copy_busy_seconds"] = stream.copy_busy_seconds;
    out["exposed_seconds"]   = stream.exposed_seconds;
    out["prefill"]           = json{{"upload_bytes", stream.prefill_upload_bytes},
                                    {"download_bytes", stream.prefill_download_bytes},
                                    {"copy_seconds", stream.prefill_copy_seconds},
                                    {"exposed_seconds", stream.prefill_exposed_seconds}};
    out["timing"]            = json{{"samples", stream.timing_samples},
                                    {"lost", stream.timing_lost}};
    // The partition controller's inputs (kv_stream_feedback.h): what fraction of copy runs were late
    // at their attention's launch, and how busy the link was in the last completed round.
    out["feedback"]          = json{{"deadline_samples", stream.deadline_samples},
                                    {"deadline_misses", stream.deadline_misses},
                                    {"busy_windows", stream.copy_busy_windows},
                                    {"copy_busy_ratio", stream.copy_busy_ratio}};
    // The byte oracle, when `NINFER_STREAM_VERIFY` is set: pages whose ring bytes were compared
    // with their host records and how many differed. A record with no checks proves nothing.
    out["verify"]            = json{{"checks", stream.verify_checks},
                                    {"mismatches", stream.verify_mismatches}};
    out["rounds_series"]     = json::array();
    for (const StreamedKvRoundSample& round : stream.round_samples) {
        json sample{{"upload_bytes", round.upload_bytes},
                    {"download_bytes", round.download_bytes},
                    {"copy_seconds", round.copy_seconds},
                    {"exposed_seconds", round.exposed_seconds},
                    {"deadline_samples", round.deadline_samples},
                    {"deadline_misses", round.deadline_misses},
                    {"copy_busy_ratio", round.copy_busy_ratio}};
        sample["layers"] = json::array();
        for (const StreamedKvLayerSample& layer : round.layers) {
            sample["layers"].push_back(json{{"layer", layer.layer},
                                            {"upload_bytes", layer.upload_bytes},
                                            {"download_bytes", layer.download_bytes},
                                            {"copy_seconds", layer.copy_seconds},
                                            {"exposed_seconds", layer.exposed_seconds}});
        }
        out["rounds_series"].push_back(std::move(sample));
    }
    return out;
}

json speculative_stats(const SpeculativeStats& speculative) {
    json out;
    out["enabled"]         = speculative.enabled;
    out["backend"]         = product::speculative_backend_name(speculative.backend);
    out["draft_window"]    = speculative.draft_window;
    out["rounds"]          = speculative.rounds;
    out["drafted_tokens"]  = speculative.drafted_tokens;
    out["accepted_tokens"] = speculative.accepted_tokens;
    out["fallback_steps"]  = speculative.fallback_steps;
    out["adaptive"]        = speculative.adaptive;
    out["window_transitions"] = speculative.window_transitions;
    out["accepted_per_position"] = speculative.accepted_per_position;
    out["rounds_per_window"]     = speculative.rounds_per_window;
    if (speculative.drafted_tokens != 0) {
        out["acceptance_rate"] = static_cast<double>(speculative.accepted_tokens) /
                                 static_cast<double>(speculative.drafted_tokens);
    }
    if (speculative.rounds != 0) {
        out["acceptance_length"] = 1.0 + static_cast<double>(speculative.accepted_tokens) /
                                            static_cast<double>(speculative.rounds);
    }
    return out;
}

json kv_transfer_stats(const RuntimeStats& stats) {
    json main;
    main["d2h_pages"] = stats.main_kv_d2h_pages;
    main["h2d_pages"] = stats.main_kv_h2d_pages;
    main["d2d_pages"] = stats.main_kv_d2d_pages;
    main["d2h_bytes"] = stats.main_kv_d2h_bytes;
    main["h2d_bytes"] = stats.main_kv_h2d_bytes;
    main["d2d_bytes"] = stats.main_kv_d2d_bytes;
    main["d2h_seconds"] = stats.main_kv_d2h_seconds;
    main["h2d_seconds"] = stats.main_kv_h2d_seconds;
    main["d2d_seconds"] = stats.main_kv_d2d_seconds;
    json backend;
    backend["d2h_pages"] = stats.backend_kv_d2h_pages;
    backend["h2d_pages"] = stats.backend_kv_h2d_pages;
    backend["d2d_pages"] = stats.backend_kv_d2d_pages;
    backend["d2h_bytes"] = stats.backend_kv_d2h_bytes;
    backend["h2d_bytes"] = stats.backend_kv_h2d_bytes;
    backend["d2d_bytes"] = stats.backend_kv_d2d_bytes;
    json state;
    state["d2h_bytes"]   = stats.state_d2h_bytes;
    state["h2d_bytes"]   = stats.state_h2d_bytes;
    state["d2d_bytes"]   = stats.state_d2d_bytes;
    state["d2h_seconds"] = stats.state_d2h_seconds;
    state["h2d_seconds"] = stats.state_h2d_seconds;
    state["d2d_seconds"] = stats.state_d2d_seconds;
    return json{{"main", main}, {"backend", backend}, {"state", state}};
}

} // namespace

std::string finish_reason_name(FinishReason reason) {
    switch (reason) {
    case FinishReason::None:
        return "none";
    case FinishReason::OutputLimit:
        return "output-limit";
    case FinishReason::ContextCapacity:
        return "context-capacity";
    case FinishReason::StopToken:
        return "stop-token";
    case FinishReason::StopString:
        return "stop-string";
    case FinishReason::Cancelled:
        return "cancelled";
    }
    return "unknown";
}

std::string kv_cache_name(KvCacheStorage storage) {
    switch (storage) {
    case KvCacheStorage::BFloat16:
        return "bf16";
    case KvCacheStorage::Int8Group64:
        return "int8-group64";
    case KvCacheStorage::Fp8E4M3Row256:
        return "fp8-e4m3-row256";
    case KvCacheStorage::RotatedInt8KeyInt4ValueGroup64:
        return "rotated-k8g64-v4g32";
    case KvCacheStorage::RotatedLloyd4KeyInt4Value:
        return "rotated-lloyd4g64-v4g32";
    case KvCacheStorage::RotatedInt4KeyInt4ValueE8:
        return "rotated-k4e8g64-v4g32";
    case KvCacheStorage::RotatedE8RootKeyInt4Value:
        return "rotated-k2e8g64-v4g32";
    case KvCacheStorage::Nvfp4Group16:
        return "nvfp4";
    case KvCacheStorage::Fp8KeyNvfp4Value:
        return "k8v4";
    }
    return "unknown";
}

std::string kv_capacity_mode_name(KvCapacityMode mode) {
    return mode == KvCapacityMode::Automatic ? "auto" : "explicit";
}

void write_run_report(const std::filesystem::path& path, const RunReport& report) {
    const GenerationResult& result = report.result;
    const MemorySummary& memory    = report.memory;
    const RuntimeStats& stats      = report.stats;

    json out;
    out["schema"] = 1;
    out["kind"]   = "ninfer-run-report";
    out["run"]    = json{{"started_utc", utc(report.started)},
                         {"finished_utc", utc(report.finished)},
                         {"wall_seconds", seconds_between(report.started, report.finished)},
                         {"argv", report.argv},
                         {"version", std::string(NINFER_VERSION)},
                         {"build_id", std::string(NINFER_BUILD_ID)}};

    out["target"] = json{
        {"model_path", report.options.artifact_path.string()},
        {"architecture", report.load.architecture},
        {"model_name", report.load.model_name},
        {"weight_formats", report.load.weight_formats},
        {"device", report.options.device},
        {"devices", report.options.devices},
        {"invariant_attention", report.options.invariant_attention},
        {"kv_stream", report.options.kv_stream},
        {"kv_stream_report", report.options.kv_stream_report ||
                                 (report.options.kv_stream &&
                                  !report.options.report_json_path.empty())},
        {"cuda_graph_requested", report.options.use_cuda_graph},
        {"max_context", report.options.max_context},
        {"max_new", report.options.max_new},
        {"prefill_chunk", report.options.prefill_chunk},
        {"kv_cache", kv_cache_name(memory.kv_cache)},
        {"kv_capacity_mode", kv_capacity_mode_name(memory.kv_capacity_mode)},
        {"kv_capacity", memory.kv_capacity},
        {"kv_capacity_page_groups", memory.kv_capacity_page_groups},
        {"kv_capacity_max_page_groups", memory.kv_capacity_max_page_groups},
        {"load_seconds", report.load.load_seconds},
        {"upload_seconds", report.load.upload_seconds},
        {"artifact_bytes_read", report.load.artifact_bytes_read},
        {"host_to_device_bytes", report.load.host_to_device_bytes},
        {"pinned_weight_bytes", report.load.pinned_weight_bytes},
    };

    out["devices"] = json::array();
    for (const LoadSummary::DeviceIdentity& device : report.load.devices) {
        out["devices"].push_back(
            json{{"index", device.index},
                 {"name", device.name},
                 {"driver_version", device.driver_version},
                 {"compute_capability",
                  std::to_string(device.compute_major) + "." + std::to_string(device.compute_minor)},
                 {"multiprocessors", device.multiprocessors},
                 {"memory_bytes", device.memory_bytes}});
    }

    out["prompt"] = json{{"tokens", result.prompt.prompt_tokens},
                         {"reused_tokens", result.reused_prompt_tokens}};
    out["sampling"] = json{{"temperature", report.sampling.temperature},
                           {"top_k", report.sampling.top_k},
                           {"top_p", report.sampling.top_p},
                           {"min_p", report.sampling.min_p},
                           {"presence_penalty", report.sampling.presence_penalty},
                           {"frequency_penalty", report.sampling.frequency_penalty},
                           {"seed", report.sampling.seed}};

    const std::size_t generated = result.generated_token_ids.size();
    const std::size_t decoded   = generated == 0 ? 0 : generated - 1;
    const double model_seconds  = result.timings.vision_seconds + result.timings.prefill_seconds +
                                 result.timings.decode_seconds;
    json generation{{"tokens", generated},
                    {"finish_reason", finish_reason_name(result.finish_reason)},
                    {"token_ids", result.generated_token_ids}};
    // The distribution record: for every content token, its log probability and those of its most
    // likely alternatives. Two runs that differ only in where the KV pages live must reproduce
    // these bit for bit, so a record compares the array itself, not a tolerance.
    if (!result.content_logprobs.empty()) {
        json logprobs = json::array();
        for (const TokenLogprob& record : result.content_logprobs) {
            json entry{{"id", record.id}, {"logprob", record.logprob}};
            entry["top_ids"]    = record.top_ids;
            entry["top_values"] = record.top_values;
            logprobs.push_back(std::move(entry));
        }
        generation["logprobs"] = std::move(logprobs);
    }
    out["generation"] = std::move(generation);
    json timings{{"prepare_seconds", result.timings.prepare_seconds},
                 {"vision_seconds", result.timings.vision_seconds},
                 {"prefill_seconds", result.timings.prefill_seconds},
                 {"decode_seconds", result.timings.decode_seconds},
                 {"total_seconds", result.timings.total_seconds},
                 {"model_seconds", model_seconds}};
    if (result.timings.prefill_seconds > 0.0) {
        timings["prefill_tokens_per_second"] =
            static_cast<double>(result.prompt.prompt_tokens) / result.timings.prefill_seconds;
    }
    if (result.timings.decode_seconds > 0.0) {
        timings["decode_tokens_per_second"] =
            static_cast<double>(decoded) / result.timings.decode_seconds;
    }
    if (model_seconds > 0.0) {
        timings["overall_tokens_per_second"] = static_cast<double>(generated) / model_seconds;
    }
    out["timings"] = std::move(timings);

    json rounds{{"count", stats.decode_rounds},
                {"sample_cap", RuntimeStats::kDecodeRoundSampleCap},
                {"samples", json::array()}};
    for (const RuntimeStats::DecodeRoundSample& sample : stats.decode_round_samples) {
        rounds["samples"].push_back(
            json{{"seconds", sample.seconds}, {"tokens", sample.tokens}});
    }
    out["decode_rounds"] = std::move(rounds);

    out["memory"]      = memory_summary(memory);
    out["stream"]      = stream_stats(stats.streamed_kv);
    out["speculative"] = speculative_stats(result.speculative);
    out["kv_transfer"] = kv_transfer_stats(stats);

    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) {
        throw std::runtime_error("cannot open the report file " + path.string());
    }
    file << out.dump(2) << '\n';
    file.flush();
    if (!file) { throw std::runtime_error("cannot write the report file " + path.string()); }
}

} // namespace ninfer::cli
