#pragma once

#include "core/host_kv_arena.h"
#include "core/kv_stream_feedback.h"
#include "core/kv_stream_layout.h"
#include "core/kv_stream_partition.h"
#include "core/kv_stream_signal.h"
#include "core/paged_kv_cache.h"
#include "models/qwen3_5/state/decoder_state.h"
#include "ninfer/observability/resources.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::models::qwen3_5 {

// The text cache's pages when they stream (docs/maintainer/kv-streaming-design.md).
//
// Pinned host memory holds every page of every full-attention layer and is the authority. The
// device pool holds those layers' pages without holding the whole context: a window over each
// layer's newest pages plus one ring of slots that every layer borrows in turn. One block table
// per layer names the slot a logical page occupies, so attention reaches a page through the table
// it already dereferences and a CUDA graph stays valid across rounds.
//
// The space owns three things the paged stores cannot express: all of the pool's slots (taken for
// the cache's lifetime, so the pool hands out nothing and lends nothing), the per-layer tables'
// contents, and the transfers that keep the host copy current. It owns no scheduling policy: when
// to move a page and which layer to fill next are the caller's.
class StreamedKVSpace {
public:
    // Takes every slot of `cache`'s pool and allocates the host authority for the whole context.
    // The cache must have been planned with this split (`PagedKVCache::stream_split`).
    // `report` adds the timing event pairs that measure the copies and the waits they cause; they
    // cost device time of their own, so a run that is being measured leaves them off.
    explicit StreamedKVSpace(PagedKVCache& cache, bool report = false);
    ~StreamedKVSpace();

    StreamedKVSpace(const StreamedKVSpace&)            = delete;
    StreamedKVSpace& operator=(const StreamedKVSpace&) = delete;
    StreamedKVSpace(StreamedKVSpace&&)                 = delete;
    StreamedKVSpace& operator=(StreamedKVSpace&&)      = delete;

    [[nodiscard]] const KVStreamSplit& split() const noexcept { return split_; }

    // Logical pages one layer reaches at the planned capacity.
    [[nodiscard]] std::uint32_t pages_per_layer() const noexcept { return pages_per_layer_; }

    [[nodiscard]] std::uint32_t layers() const noexcept { return layers_; }

    // Layers' streamed pages the ring holds at once: how far ahead of the attending layer a copy
    // may run.
    [[nodiscard]] std::uint32_t ring_depth() const noexcept { return split_.ring_depth; }

    // Pages the tables describe, and where the resident window currently starts.
    [[nodiscard]] std::uint32_t active_pages() const noexcept { return active_; }

    [[nodiscard]] std::uint32_t window_start() const noexcept {
        return active_ > split_.resident_pages ? active_ - split_.resident_pages : 0U;
    }

    // Host pages of one layer: page `p` is record `p` of this view, so a caller can copy a run of
    // them with a plain 2D transfer or point the append Ops at them. Every layer has its own run:
    // two layers' page `p` are different bytes and never share a record.
    [[nodiscard]] HostKVAllocationView host_pages(std::uint32_t layer) noexcept;
    [[nodiscard]] HostKVAllocationConstView host_pages(std::uint32_t layer) const noexcept;

    // Device slot handles, indexed by slot number.
    [[nodiscard]] std::uint32_t slot_count() const noexcept { return slot_count_; }

    // Brings the tables to `active_pages` logical pages of the context, creating the pages in
    // between: each new page's table entry points at its window slot, and each page the window
    // drops first moves to the host and then points at its ring slot. Both are ordered on
    // `streams`, so a page's slot is never reused before its bytes reached the host.
    //
    // Growing is the only transition a served request makes; `reset` is how a new request clears
    // the tables and the frontier.
    void advance(std::uint32_t active_pages, RankStreams streams);

    // Forgets the context: the frontier returns to zero and the next request republishes each
    // entry as its pages appear. Nothing is copied, because those pages are written again.
    void reset() noexcept {
        active_ = 0;
        // The ring holds the previous request's pages, so nothing may be read as if it were this
        // request's.
        region_layer_.assign(region_layer_.size(), kNoLayer);
        region_pages_.assign(region_pages_.size(), 0);
    }

    // Copies layer `layer`'s streamed pages (`[0, window_start)`) from their host pages into the
    // ring, on `copy`. The copies are asynchronous and land in the ring slots the tables name; the
    // caller records a ready event afterwards and has the consumer wait on it.
    //
    // The ring is shared, so a round copies each layer into its own region of it: the region the
    // layer `ring_depth` steps back occupied is the one this copy overwrites, which is why the
    // caller orders it after that layer's consumption rather than the previous layer's.
    //
    // A region that already holds this layer's pages up to the frontier only receives the pages it
    // lacks. That is what lets the last layers of a round copy the *next* round's first layers into
    // their regions without those pages being copied twice.
    void upload(std::uint32_t layer, cudaStream_t copy);

    // The pages a decode round feeds one layer at a time: the call's keys are launched in frames,
    // so its copies are issued in frame-sized runs instead of one layer-sized one, and a run can
    // fill a gap between two launches that a whole layer's copy could not (design 5.13.2).
    //
    // `upload_frame` moves the streamed pages of `[first_page, first_page + page_count)` into their
    // ring slots, for the frame index the caller's schedule uses: it waits for the layer that
    // shared those slots `ring_depth` layers back, records `ready_frame`, and returns the pages it
    // moved (zero when the range is entirely in the window). The caller may split a frame's range
    // across several calls; it records what it waits on through `reserve_frame`/`seal_frame`.
    static constexpr std::uint32_t kFrameSlots = 8;
    std::uint32_t upload_frame(std::uint32_t layer, std::uint32_t frame, std::int32_t first_page,
                               std::int32_t page_count, cudaStream_t copy, bool seal = true);

    // Records `ready_frame` without copying, which is how a frame whose pages are all resident gets
    // an event the consumer can wait on.
    void seal_frame(std::uint32_t layer, std::uint32_t frame, cudaStream_t copy);

    // Waits on `frame`'s ready event on the compute stream, measuring how much of the wait was
    // copy time the round could not hide.
    void wait_ready_frame(std::uint32_t layer, std::uint32_t frame, cudaStream_t compute);

    // The event the consumer records once a frame's launch has read the ring.
    [[nodiscard]] cudaEvent_t consumed_frame_event(std::uint32_t layer,
                                                   std::uint32_t frame) const;

    // First slot of the ring region layer `layer`'s streamed pages occupy.
    [[nodiscard]] std::uint32_t ring_region(std::uint32_t layer) const noexcept {
        return ring_base_ + (layer % split_.ring_depth) *
                                kv_stream_ring_extent(split_, layers_);
    }

    // The physical space on each side of the link this cache holds for its lifetime.
    [[nodiscard]] std::uint64_t host_bytes() const noexcept {
        return page_bytes_ * pages_per_layer_ * layers_;
    }
    [[nodiscard]] std::uint64_t device_bytes() const noexcept {
        return page_bytes_ * slot_count_;
    }

    // The per-layer events a round orders its copies and its consumers with. `ready(layer)` is
    // recorded by `upload`; `consumed(layer)` is recorded by the consumer once that layer's
    // attention has read the ring. `dropped(layer)` covers a round's host copies of the pages that
    // page's window dropped, so the upload that reads them waits for their bytes to arrive.
    [[nodiscard]] cudaEvent_t ready_event(std::uint32_t layer) const;
    [[nodiscard]] cudaEvent_t consumed_event(std::uint32_t layer) const;
    [[nodiscard]] cudaEvent_t dropped_event(std::uint32_t layer) const;

    // One decode round as the link saw it. One entry per started round, oldest first, even when
    // nothing moved: index `i` is decode round `i` of RuntimeStats::decode_round_samples.
    using RoundSample = StreamedKvRoundSample;

    // What one request's streaming asked of the PCIe link, and what the compute stream waited for.
    // `exposed_seconds` is the number that decides whether scheduling work can still buy anything:
    // it counts the time the compute stream spent on a `wait_ready` whose copy had not landed.
    // Byte counts are measured page moves, not a formula over the frontier, so a prefetched page
    // the region already holds is not counted twice.
    struct TransferStats {
        std::uint32_t resident_pages  = 0;
        std::uint32_t streamed_pages  = 0;
        std::uint32_t layers          = 0;
        std::uint32_t ring_depth      = 0;
        std::uint32_t pages_per_layer = 0;
        std::uint64_t page_bytes      = 0;
        std::uint64_t host_bytes      = 0;
        std::uint64_t device_bytes    = 0;
        std::uint64_t rounds          = 0;
        std::uint64_t upload_bytes    = 0;
        std::uint64_t download_bytes  = 0;
        double copy_busy_seconds      = 0.0;
        double exposed_seconds        = 0.0;
        // The prompt's traffic, which has no decode round of its own.
        std::uint64_t prefill_upload_bytes   = 0;
        std::uint64_t prefill_download_bytes = 0;
        double prefill_copy_seconds          = 0.0;
        double prefill_exposed_seconds       = 0.0;
        std::uint64_t timing_samples         = 0;
        std::uint64_t timing_lost            = 0;
        // The two signals the partition controller reads (kv_stream_feedback.h): how many copy runs
        // were still not ready when their attention launched, and how much of the round the copy
        // engine was busy. `copy_busy_ratio` is the last completed window's estimate, not an
        // average: the controller decides per window.
        std::uint64_t deadline_samples    = 0;
        std::uint64_t deadline_misses     = 0;
        std::uint64_t copy_busy_windows   = 0;
        double copy_busy_ratio            = 0.0;
        std::vector<RoundSample> round_samples;
    };

    // Starts a decode round: the previous round's window is closed against `compute`, so its
    // deadline counters and copy-busy estimate are read from the device's own record, and the new
    // round opens its window (kv_stream_feedback.h).
    void begin_round(cudaStream_t compute);

    // The controller's decision for this round, applied when it moves the boundary (port of the
    // reference implementation's `kv_stream_adapt` + `reconfigure`/`repartition`): the ring this
    // round's window needed, how late its copies were, and how busy the link was decide whether the
    // window may grow. Returns true when the pool moved.
    bool evaluate_partition(RankStreams streams);

    // Moves the boundary between the resident window and the ring inside this space's fixed pool.
    // Every slot is re-addressed, so the pages that change sides migrate once - a page leaving the
    // window is written back to its host record first, a page entering it is read back from there -
    // and the block tables are republished. Rare by design: the controller rate-limits it.
    void repartition(const KVStreamSplit& next, RankStreams streams);

    // Waits on `layer`'s ready event on the compute stream and measures how much of that wait was
    // real, which is the copy time this round could not hide.
    void wait_ready(std::uint32_t layer, cudaStream_t compute);

    // Reads every timing pair the device has finished, closes the round in flight and returns the
    // series. Safe to call repeatedly: a pair is counted once.
    [[nodiscard]] TransferStats transfer_stats();

private:
    void publish_slot(std::uint32_t layer, std::uint32_t page, std::int32_t slot,
                      RankStreams streams);
    // Reads every complete pair of timing events into the round in flight (or the prompt's
    // accumulators before the first round). A pair whose events are not finished is left pending;
    // the copy that overwrites it counts as `timing_lost`.
    void collect_timings();
    [[nodiscard]] std::optional<double> take_timing(cudaEvent_t begin, cudaEvent_t end,
                                                    std::uint8_t& pending);
    // Accounts for a pair a re-recorded copy is about to overwrite: its time if the device
    // finished it, otherwise one lost sample.
    void close_timing(std::uint8_t& pending, cudaEvent_t begin, cudaEvent_t end,
                      std::uint32_t layer, bool copy);
    void add_copy_seconds(std::uint32_t layer, double seconds);
    void add_exposed_seconds(std::uint32_t layer, double seconds);
    void reset_round();
    [[nodiscard]] RoundSample snapshot_round() const;
    void close_round();
    void add_upload(std::uint32_t layer, std::uint64_t bytes);
    void add_download(std::uint32_t layer, std::uint64_t bytes);
    // Closes the round's feedback window if the device finished it, and opens nothing: the caller
    // opens the next one at its first round boundary.
    [[nodiscard]] bool close_signal_window() noexcept;

    PagedKVCache* cache_               = nullptr;
    // The host authority's pages, allocated for this cache alone: the retention arena belongs to a
    // cache that checkpoints, which a streaming cache refuses to be.
    std::unique_ptr<HostKVArena> arena_;
    KVStreamSplit split_;
    std::uint32_t layers_              = 0;
    std::uint32_t pages_per_layer_     = 0;
    std::uint32_t ring_base_           = 0;
    std::uint32_t slot_count_          = 0;
    std::size_t page_bytes_            = 0;
    HostKVPageLayout host_layout_;
    HostKVAllocation host_;
    std::vector<cudaEvent_t> ready_;    std::vector<cudaEvent_t> consumed_;
    std::vector<cudaEvent_t> dropped_;
    // One ready/consumed pair per (layer, frame slot): a frame's copy and the launch that reads it
    // are ordered through these, and a slot is reused once `ring_depth` layers have consumed it.
    std::vector<cudaEvent_t> ready_frame_;
    std::vector<cudaEvent_t> consumed_frame_;
    std::vector<cudaEvent_t> frame_copy_begin_;
    std::vector<cudaEvent_t> frame_copy_end_;
    std::vector<cudaEvent_t> frame_wait_begin_;
    std::vector<cudaEvent_t> frame_wait_end_;
    std::vector<std::uint8_t> frame_copy_pending_;
    std::vector<std::uint8_t> frame_wait_pending_;
    void add_frame_copy_seconds(std::uint32_t layer, double seconds);
    void add_frame_exposed_seconds(std::uint32_t layer, double seconds);
    // Timing pairs, one per layer for each of the two phases of a round: the copy on the copy
    // stream and the wait for it on the compute stream.
    std::vector<cudaEvent_t> copy_begin_;
    std::vector<cudaEvent_t> copy_end_;
    std::vector<cudaEvent_t> wait_begin_;
    std::vector<cudaEvent_t> wait_end_;
    std::vector<std::uint8_t> copy_pending_;
    std::vector<std::uint8_t> wait_pending_;
    bool report_                 = false;
    std::uint64_t rounds_        = 0;
    std::uint64_t upload_bytes_  = 0;
    std::uint64_t download_bytes_ = 0;
    double copy_seconds_         = 0.0;
    double exposed_seconds_      = 0.0;
    std::uint64_t prefill_upload_bytes_   = 0;
    std::uint64_t prefill_download_bytes_ = 0;
    double prefill_copy_seconds_          = 0.0;
    double prefill_exposed_seconds_       = 0.0;
    std::uint64_t timing_samples_         = 0;
    std::uint64_t timing_lost_            = 0;
    // The round in flight. `layers` is pre-sized to `layers_` so a sample costs no allocation;
    // closing the round keeps only the layers that moved something.
    RoundSample round_;
    // One sample per started round, oldest first, capped: a run past the cap keeps exact totals and
    // `rounds_` still tells how many rounds there were. The cap matches
    // RuntimeStats::kDecodeRoundSampleCap so index `i` stays the same round in both series.
    static constexpr std::size_t kRoundSampleCap = 2048;
    std::vector<RoundSample> round_samples_;
    // The streamed pages one round moved, recorded when the round started: the frontier resets with
    // the request, so the live value is not what a finished request moved.
    std::uint32_t round_streamed_pages_ = 0;
    std::uint32_t active_ = 0;
    // What each region of the ring holds: the layer that last copied into it and how many of that
    // layer's pages from page 0 it has. A copy of the same layer into the same region moves only
    // the pages it does not already hold. `kNoLayer` means the region holds nothing (a new
    // request, or a region no layer has reached yet).
    static constexpr std::uint32_t kNoLayer =
        std::numeric_limits<std::uint32_t>::max();
    std::vector<std::uint32_t> region_layer_;
    std::vector<std::uint32_t> region_pages_;
    // The deadline flags, the counters and the two window events the controller's inputs come from.
    std::unique_ptr<KVStreamSignal> signal_;
    KVStreamDeadlineCounters signal_base_{};
    std::uint64_t signal_uploads_     = 0;
    bool signal_window_open_          = false;
    bool signal_end_recorded_         = false;
    std::uint64_t deadline_samples_   = 0;
    std::uint64_t deadline_misses_    = 0;
    std::uint64_t copy_busy_windows_  = 0;
    double copy_busy_ratio_           = 0.0;
    // The last closed window's signals, which is what the controller evaluates: a window still in
    // flight is not a measurement yet.
    std::uint64_t window_samples_ = 0;
    std::uint64_t window_misses_  = 0;
    double window_busy_           = 0.0;
    bool window_measured_         = false;
    // The controller's state: the ring may never shrink below the size the cache was planned with
    // (the reference sets its minimum from the initial ring), and a repartition costs a table
    // rebuild, so it runs at most once per cooldown.
    std::uint32_t minimum_ring_pages_           = 0;
    std::uint32_t evaluations_since_repartition_ = UINT32_MAX;
    std::uint32_t starved_evaluations_          = 0;
    std::uint32_t overprovisioned_evaluations_  = 0;
    bool entering_decode_layout_                = true;
};

} // namespace ninfer::models::qwen3_5
