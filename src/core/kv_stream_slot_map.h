#pragma once

#include <cstdint>
#include <stdexcept>

namespace ninfer {

// The ring as a flat pool of slots that any layer may borrow, which is what lets the ring be sized
// by the work it must cover instead of by whole layers
// (docs/maintainer/kv-streaming-fidelity.md S5; the reference implementation's `free_slots` /
// `slot_request` allocator in fattn.cu, reduced to what a single-request engine needs).
//
// The reference allocates individual pages per request and coalesces runs that happen to be
// adjacent; this engine copies and consumes in fixed runs of `pages_per_run` pages, so the pool hands
// out aligned runs instead: the slot of page `i` of a run is `begin + i`, the run never straddles the
// pool's end, and the consumer's span is exactly the run it waited for. `ring_pages` is rounded down
// to a multiple of the run so that property holds for every run.
//
// Reuse is FIFO by construction: the cursor only moves forward, so the run handed out `ring_pages /
// pages_per_run` runs ago is the one being handed out now, and it is free as long as its consumer
// finished one run earlier than the producer arrives.
class KVStreamSlotMap {
public:
    struct Run {
        std::uint32_t begin = 0;
        std::uint32_t count = 0;
    };

    explicit KVStreamSlotMap(std::uint32_t ring_pages, std::uint32_t pages_per_run)
        : pages_per_run_(pages_per_run) {
        if (pages_per_run == 0) {
            throw std::invalid_argument("A KV stream slot map needs a non-zero run");
        }
        slots_ = ring_pages - ring_pages % pages_per_run;
    }

    // Slots the pool actually hands out: `ring_pages` rounded down to a whole run.
    [[nodiscard]] std::uint32_t slots() const noexcept { return slots_; }

    [[nodiscard]] std::uint32_t pages_per_run() const noexcept { return pages_per_run_; }

    [[nodiscard]] std::uint32_t runs() const noexcept {
        return pages_per_run_ == 0 ? 0 : slots_ / pages_per_run_;
    }

    // The next run, and the cursor's move past it. A pool smaller than one run cannot hand anything
    // out; the caller then has no ring to stream through.
    [[nodiscard]] Run next() noexcept {
        if (slots_ == 0) { return Run{}; }
        const Run run{cursor_, pages_per_run_};
        cursor_ = (cursor_ + pages_per_run_) % slots_;
        return run;
    }

    void reset() noexcept { cursor_ = 0; }

    [[nodiscard]] std::uint32_t cursor() const noexcept { return cursor_; }

private:
    std::uint32_t slots_        = 0;
    std::uint32_t pages_per_run_ = 0;
    std::uint32_t cursor_       = 0;
};

} // namespace ninfer
