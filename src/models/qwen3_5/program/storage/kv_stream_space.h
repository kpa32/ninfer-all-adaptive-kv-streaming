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

// The text cache's pages when they stream (docs/maintainer/kv-streaming-design.md,
// docs/maintainer/kv-streaming-fidelity.md).
//
// Pinned host memory holds every page of every full-attention layer and is the authority. The
// device pool holds those layers' pages without holding the whole context: a window over each
// layer's newest pages plus one flat ring that every layer borrows in runs of `run_pages` pages.
// One block table per layer names the slot a logical page occupies, so attention reaches a page
// through the table it already dereferences and a CUDA graph stays valid across rounds.
//
// The space owns three things the paged stores cannot express: all of the pool's slots (taken for
// the cache's lifetime, so the pool hands out nothing and lends nothing), the per-layer tables'
// contents, and the transfers that keep the host copy current. It owns no scheduling policy beyond
// the ring's rotation: which layer copies next is the caller's.
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

    // Runs one layer's streamed pages take at the current context, and the pages a run covers.
    [[nodiscard]] std::uint32_t runs_per_layer() const noexcept {
        return kv_stream_runs_per_layer(active_, split_);
    }    [[nodiscard]] std::uint32_t run_pages() const noexcept { return split_.run_pages; }
    // Runs the ring holds, which is how far ahead of the attending layer a copy may run.
    [[nodiscard]] std::uint32_t ring_runs() const noexcept { return ring_runs_; }

    // The layout this space was built with, bumped whenever the resident/ring boundary moves: a
    // measurement taken in one layout says nothing about another, so the caller resets its tuners on
    // a change (kv_stream_span_tuner.h).
    [[nodiscard]] std::uint32_t layout_generation() const noexcept { return layout_generation_; }

    // The device time of the last closed decode round's window, zero until one completes. The span
    // tuner compares pipelines by it.
    [[nodiscard]] double last_window_seconds() const noexcept { return window_seconds_; }

    // Pages the tables describe, and where the resident window currently starts.
    [[nodiscard]] std::uint32_t active_pages() const noexcept { return active_; }

    // The narrowest window any streaming layer keeps: a round is described by its largest streamed
    // extent, so the reported numbers use the layer that streams most.
    [[nodiscard]] std::uint32_t window_start() const noexcept {
        return kv_stream_window_start(active_, split_, 0);
    }

    // The same for one layer, whose window may be wider when a round concentrates its streams.
    [[nodiscard]] std::uint32_t window_start(std::uint32_t layer) const noexcept {
        return kv_stream_window_start(active_, split_, layer);
    }

    // Pages any layer streams at the current context - the widest streamed extent, which is what the
    // ring has to hold.
    [[nodiscard]] std::uint32_t max_streamed_pages() const noexcept {
        return max_streamed_pages(split_);
    }

    [[nodiscard]] std::uint32_t max_streamed_pages(const KVStreamSplit& of) const noexcept {
        std::uint32_t most = 0;
        for (std::uint32_t layer = 0; layer < layers_; ++layer) {
            const std::uint32_t streamed = kv_stream_window_start(active_, of, layer);
            if (streamed > most) { most = streamed; }
        }
        return most;
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
    // drops first moves to the host. Both are ordered on `streams`, so a page's slot is never reused
    // before its bytes reached the host.
    //
    // Growing is the only transition a served request makes; `reset` is how a new request clears
    // the tables and the frontier.
    void advance(std::uint32_t active_pages, RankStreams streams);

    // Forgets the context: the frontier returns to zero and the next request republishes each
    // entry as its pages appear. Nothing is copied, because those pages are written again.
    void reset() noexcept {
        active_ = 0;
        run_cursor_ = 0;
        run_used_.assign(run_used_.size(), 0);
        run_owner_.assign(run_owner_.size(), kNoLayer);
        run_layer_.assign(run_layer_.size(), kNoLayer);
        layer_runs_.assign(layer_runs_.size(), 0);
    }

    // Copies layer `layer`'s streamed pages (`[0, window_start)`) from their host pages into the
    // ring, on `copy`, in runs of `run_pages` pages, and republishes that layer's streamed table
    // entries in one copy. Returns the runs it issued (zero when every page is resident).
    //
    // A run's copy waits on the consumed event of the run it takes, so a ring that holds one extent
    // serialises the layers and a ring that holds one extent plus a margin overlaps them
    // (kv_stream_layout.h). The caller records `ready` per run itself; `consume` is what frees it.
    std::uint32_t upload(std::uint32_t layer, cudaStream_t copy);

    // Waits on the layer's last run on the compute stream and samples one deadline for the layer: the
    // runs land in order on one copy stream, so the last one implies the rest.
    void wait_ready(std::uint32_t layer, cudaStream_t compute);

    // The same for one run, which is what a consumer that dispatches a run at a time waits on.
    void wait_run(std::uint32_t layer, std::uint32_t run, cudaStream_t compute);
    // A launch reads a contiguous page range, which may cover several runs or part of one: the range
    // form is what keeps the hook and the copies in agreement when a run is not a whole number of
    // attention spans (text.h, StreamFrameFeed).
    void wait_run_range(std::uint32_t layer, std::uint32_t first_run, std::uint32_t last_run,
                        cudaStream_t compute);

    // Frees the layer's runs: the consumer records this once the attention that read them finished.
    void consume(std::uint32_t layer, cudaStream_t compute);
    void consume_run(std::uint32_t layer, std::uint32_t run, cudaStream_t compute);
    void consume_run_range(std::uint32_t layer, std::uint32_t first_run, std::uint32_t last_run,
                           cudaStream_t compute);

    // The ring run a layer's first run takes this round, and the slot a page is named by now.
    [[nodiscard]] std::uint32_t first_run(std::uint32_t layer) const;
    [[nodiscard]] std::uint32_t slot_of_page(std::uint32_t layer, std::uint32_t page) const;

    // The events a caller orders its own waits with, when it dispatches run by run.
    [[nodiscard]] cudaEvent_t ready_run_event(std::uint32_t layer, std::uint32_t run) const;
    [[nodiscard]] cudaEvent_t consumed_run_event(std::uint32_t layer, std::uint32_t run) const;

    // The physical space on each side of the link this cache holds for its lifetime.
    [[nodiscard]] std::uint64_t host_bytes() const noexcept {
        return page_bytes_ * pages_per_layer_ * layers_;
    }
    [[nodiscard]] std::uint64_t device_bytes() const noexcept {
        return page_bytes_ * slot_count_;
    }

    // The per-layer events a round orders its copies and its consumers with. `ready(layer)` is
    // recorded by `upload`; `consumed(layer)` is recorded by `consume`. `dropped(layer)` covers a
    // round's host copies of the pages that page's window dropped, so the upload that reads them
    // waits for their bytes to arrive.
    [[nodiscard]] cudaEvent_t ready_event(std::uint32_t layer) const;
    [[nodiscard]] cudaEvent_t consumed_event(std::uint32_t layer) const;
    [[nodiscard]] cudaEvent_t dropped_event(std::uint32_t layer) const;

    // One decode round as the link saw it. One entry per started round, oldest first, even when
    // nothing moved: index `i` is decode round `i` of RuntimeStats::decode_round_samples.
    using RoundSample = StreamedKvRoundSample;

    // What one request's streaming asked of the PCIe link, and what the compute stream waited for.
    // `exposed_seconds` counts the time the compute stream spent on a `wait_ready` whose copy had not
    // landed. Byte counts are measured page moves, not a formula over the frontier, so a prefetched
    // page the ring already holds is not counted twice.
    struct TransferStats {
        std::uint32_t resident_pages  = 0;
        std::uint32_t streamed_pages  = 0;
        std::uint32_t layers          = 0;
        std::uint32_t ring_runs       = 0;
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
        std::uint64_t deadline_samples   = 0;
        std::uint64_t deadline_misses    = 0;
        std::uint64_t copy_busy_windows  = 0;
        double copy_busy_ratio           = 0.0;
        std::vector<RoundSample> round_samples;
    };

    // Starts a decode round: the previous round's window is closed against `compute`, so its
    // deadline counters and copy-busy estimate are read from the device's own record, and the new
    // round opens its window (kv_stream_feedback.h). The partition is re-evaluated from that
    // measurement before the round's first layer.
    void begin_round(cudaStream_t compute);

    // The controller's decision for this round, applied when it moves the boundary (port of the
    // reference implementation's `kv_stream_adapt` + `reconfigure`/`repartition`). Returns true when
    // the pool moved.
    bool evaluate_partition(RankStreams streams);

    // Moves the boundary between the resident window and the ring inside this space's fixed pool.
    // Every slot is re-addressed, so the pages that change sides migrate once - a page leaving the
    // window is written back to its host record first, a page entering it is read back from there -
    // and the block tables are republished. Rare by design: the controller rate-limits it.
    void repartition(const KVStreamSplit& next, RankStreams streams);

    // Reads every timing pair the device has finished, closes the round in flight and returns the
    // series. Safe to call repeatedly: a pair is counted once.
    [[nodiscard]] TransferStats transfer_stats();

private:
    void verify_streamed_page(std::uint32_t layer, std::uint32_t streamed);
    void verify_after_attention(std::uint32_t layer);
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
    std::uint32_t ring_runs_           = 0;
    std::uint32_t slot_count_          = 0;
    std::size_t page_bytes_            = 0;
    HostKVPageLayout host_layout_;
    HostKVAllocation host_;
    std::vector<cudaEvent_t> ready_;    std::vector<cudaEvent_t> consumed_;
    std::vector<cudaEvent_t> dropped_;
    // One ready/consumed pair per ring run: a run's copy and the span that reads it are ordered
    // through these, and a run is reused once its previous owner consumed it.
    std::vector<cudaEvent_t> ready_run_;
    std::vector<cudaEvent_t> consumed_run_;
    std::vector<std::uint8_t> run_used_;
    std::vector<std::uint32_t> run_owner_;
    // Which layer a run currently holds, so a consumer can tell whose measurement it is holding.
    std::vector<std::uint32_t> run_layer_;
    std::vector<std::uint32_t> layer_first_run_;
    std::vector<std::uint32_t> layer_runs_;
    // The slots the streamed prefix is named by this round, published in one copy per layer.
    std::vector<std::int32_t> stream_slots_;
    std::uint32_t run_cursor_ = 0;
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
    // The deadline flags, the counters and the two window events the controller's inputs come from.
    std::unique_ptr<KVStreamSignal> signal_;
    KVStreamDeadlineCounters signal_base_{};
    std::uint64_t signal_uploads_    = 0;
    bool signal_window_open_         = false;
    bool signal_end_recorded_        = false;
    std::uint64_t deadline_samples_  = 0;
    std::uint64_t deadline_misses_   = 0;
    std::uint64_t copy_busy_windows_ = 0;
    double copy_busy_ratio_          = 0.0;
    // The last closed window's signals, which is what the controller evaluates.
    std::uint64_t window_samples_ = 0;
    std::uint64_t window_misses_  = 0;
    double window_busy_           = 0.0;
    double window_seconds_        = 0.0;
    std::uint32_t layout_generation_ = 0;
    bool window_measured_         = false;
    // The controller's state: the ring may never shrink below the size the cache was planned with
    // (the reference sets its minimum from the initial ring), and a repartition costs a table rebuild,
    // so it runs at most once per cooldown.
    std::uint32_t minimum_ring_pages_            = 0;
    std::uint32_t evaluations_since_repartition_ = UINT32_MAX;
    std::uint32_t starved_evaluations_           = 0;
    std::uint32_t overprovisioned_evaluations_   = 0;
    // The `NINFER_STREAM_VERIFY` diagnostic: how many streamed pages it still compares, and the one-page
    // pinned staging area it reads a ring slot into.
    int verify_budget_ = 20000;
    int verify_after_budget_ = 64;
    std::unique_ptr<HostKVArena> verify_arena_;
    std::optional<HostKVAllocation> verify_host_;
    static constexpr std::uint32_t kNoLayer = std::numeric_limits<std::uint32_t>::max();
};

} // namespace ninfer::models::qwen3_5
