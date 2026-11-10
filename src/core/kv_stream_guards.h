#pragma once

#include <cstdint>

namespace ninfer {

// What rules a request out of the streaming KV cache (docs/maintainer/kv-streaming-design.md).
//
// A streamed pool holds *slots*, not contexts: the device keeps each full-attention layer's newest
// pages plus one ring that every layer borrows, and the block table names whichever slot a logical
// page currently occupies. Everything that would make a device page stop meaning "this layer's whole
// context" therefore rules the request out, and the reason is named here rather than in a branch of
// the startup planner: the same table is what `tests/test_kv_stream_guards.cpp` checks, so a new
// exclusion cannot quietly become a message nobody tests and an accepted combination cannot quietly
// lose its guard.
enum class KVStreamRejection : std::uint8_t {
    // The request is servable with the streaming cache, or it does not ask for one.
    None = 0,
    // The pool is sized from measured free memory; an explicit token count would size it by arithmetic
    // that cannot see the ring.
    CapacityNotAutomatic,
    // A second sequence would have to share the ring, whose slots are claimed per layer per round.
    ManySequences,
    // Pipeline stages and tensor-parallel ranks split every layer's KV over the devices, while a
    // streaming pool keeps one layer's whole context on one device.
    ManyDevices,
    // Retained context and checkpoints reason about device pages as whole-context pages, which a
    // streamed pool does not have. A streamed run retains its own continuation instead: the pinned
    // host authority already holds every page, so the checkpoint costs a StateImage, not a second
    // KV copy (see `streamed_retention`).
    RetainedContext,
    // The timing report exists to measure the streaming loop.
    ReportWithoutStream,
    // A retained streamed context is a host StateImage beside the pinned host authority; a device
    // state slot is a second device image beside the window, which the retain path does not manage.
    RetainedContextDeviceSlot,
    // The endpoint has to live somewhere, and for a streamed run that somewhere is the host shelf.
    RetainedContextNoHostSlot,
};

// The request as the planner sees it, copied out of the CLI/Engine options so this stays a pure
// function of plain values.
struct KVStreamRequest {
    bool enabled             = false;
    bool report              = false;
    bool capacity_automatic  = false;
    std::uint32_t sequences  = 1;
    std::size_t devices      = 1;
    bool tensor_parallel     = false;
    // The request enables a context cache at all.
    bool retained_context    = false;
    // That cache is the streamed retention: the streamed arena is the host KV tier, so the
    // checkpoint stores identity and state and leaves the pages where they are.
    bool streamed_retention  = false;
    // Where the retention's endpoint StateImage lives. A streamed retention keeps it on the host
    // shelf beside the arena; a device state slot would place a third device image beside the
    // window and rotate the lane's read/write pair against the page it is writing.
    std::uint32_t device_state_slots = 0;
    std::uint32_t host_state_slots   = 0;
};

// Why `request` cannot run with the streaming KV cache, in the order a user meets the conditions.
[[nodiscard]] constexpr KVStreamRejection kv_stream_rejection(
    const KVStreamRequest& request) noexcept {
    if (request.report && !request.enabled) { return KVStreamRejection::ReportWithoutStream; }
    if (!request.enabled) { return KVStreamRejection::None; }
    if (!request.capacity_automatic) { return KVStreamRejection::CapacityNotAutomatic; }
    if (request.sequences != 1) { return KVStreamRejection::ManySequences; }
    if (request.devices != 1 || request.tensor_parallel) { return KVStreamRejection::ManyDevices; }
    if (request.retained_context && !request.streamed_retention) {
        return KVStreamRejection::RetainedContext;
    }
    if (request.retained_context && request.device_state_slots != 0) {
        return KVStreamRejection::RetainedContextDeviceSlot;
    }
    if (request.retained_context && request.host_state_slots == 0) {
        return KVStreamRejection::RetainedContextNoHostSlot;
    }
    return KVStreamRejection::None;
}

// The message the CLI prints. Every accepted request has no message; every rejected one names the flag
// that fixes it, because these are the only paths a user can hit.
[[nodiscard]] constexpr const char* kv_stream_rejection_message(KVStreamRejection rejection) noexcept {
    switch (rejection) {
    case KVStreamRejection::None:
        return "";
    case KVStreamRejection::CapacityNotAutomatic:
        return "--kv-stream sizes its KV pool from free memory: pass --kv-capacity auto";
    case KVStreamRejection::ManySequences:
        return "--kv-stream serves one sequence: set --max-concurrency 1";
    case KVStreamRejection::ManyDevices:
        return "--kv-stream keeps the KV cache on one device";
    case KVStreamRejection::RetainedContext:
        return "--kv-stream cannot checkpoint or reuse retained context (--no-prefix-reuse)";
    case KVStreamRejection::RetainedContextDeviceSlot:
        return "--kv-stream retains its endpoint on the host shelf: pass --device-state-slots 0";
    case KVStreamRejection::RetainedContextNoHostSlot:
        return "--kv-stream retains its endpoint StateImage: give it one host state slot "
               "(--host-state-slots 1)";
    case KVStreamRejection::ReportWithoutStream:
        return "--kv-stream-report measures --kv-stream: pass --kv-stream";
    }
    return "";
}

} // namespace ninfer
