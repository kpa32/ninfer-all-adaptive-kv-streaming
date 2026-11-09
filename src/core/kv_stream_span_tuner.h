#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace ninfer {

// Chooses between an ordinary coalesced streamed-attention launch and a bounded-span pipeline, using
// end-to-end decode round timings (ported from the reference implementation's
// `ggml_cuda_kv_stream_span_tuner`). The tuner is reset whenever the resident/ring layout changes,
// because a different layout is a different trade.
//
// Why this has to be measured instead of derived: the bounded pipeline lets the consumer start as
// soon as the first run of pages lands, which is what makes a ring of barely more than one layer's
// extent pay. It also turns one launch per layer into several and can leave the device's waves
// ragged. Which effect wins depends on the card, the cache type and the ring's size - the reference
// decides it the same way, and only after sixteen samples of each, with a half-percent margin.
class KVStreamSpanTuner {
public:
    explicit KVStreamSpanTuner(std::uint32_t trial_samples = 16,
                               double minimum_relative_gain = 0.005,
                               std::uint32_t warmup_samples = 1)
        : trial_samples_(std::max<std::uint32_t>(trial_samples, 1)),
          minimum_relative_gain_(std::isfinite(minimum_relative_gain)
                                     ? std::clamp(minimum_relative_gain, 0.0, 0.999)
                                     : 0.005),
          warmup_samples_(warmup_samples) {}

    void reset() noexcept {
        bounded_          = false;
        selected_         = false;
        unbounded_samples_ = 0;
        bounded_samples_  = 0;
        unbounded_warmups_ = 0;
        bounded_warmups_  = 0;
        unbounded_ms_     = 0.0;
        bounded_ms_       = 0.0;
    }

    // One finished decode round: `elapsed_ms` of the whole round, whether the round streamed at all
    // (a round with nothing to copy measures nothing about the pipeline), and which pipeline it ran.
    void observe(double elapsed_ms, bool streamed, bool sample_was_bounded) noexcept {
        if (selected_ || !streamed || !std::isfinite(elapsed_ms) || elapsed_ms <= 0.0) { return; }

        std::uint32_t& warmups = sample_was_bounded ? bounded_warmups_ : unbounded_warmups_;
        if (warmups < warmup_samples_) {
            ++warmups;
            return;
        }
        if (sample_was_bounded) {
            bounded_ms_ += elapsed_ms;
            ++bounded_samples_;
        } else {
            unbounded_ms_ += elapsed_ms;
            ++unbounded_samples_;
        }

        // Both pipelines are measured before anything is chosen: an unbounded round is the baseline
        // the bounded one has to beat, so the first rounds simply collect it.
        if (unbounded_samples_ < trial_samples_) {
            bounded_ = false;
            return;
        }
        if (bounded_samples_ < trial_samples_) {
            bounded_ = true;
            return;
        }
        const double unbounded_average = unbounded_ms_ / static_cast<double>(unbounded_samples_);
        const double bounded_average   = bounded_ms_ / static_cast<double>(bounded_samples_);
        bounded_  = bounded_average < unbounded_average * (1.0 - minimum_relative_gain_);
        selected_ = true;
    }

    [[nodiscard]] bool use_bounded() const noexcept { return bounded_; }
    [[nodiscard]] bool selected() const noexcept { return selected_; }
    // True while the tuner still wants rounds of both pipelines: the caller alternates with it.
    [[nodiscard]] bool collecting() const noexcept {
        return !selected_ && (unbounded_samples_ < trial_samples_ || bounded_);
    }
    [[nodiscard]] double unbounded_average_ms() const noexcept {
        return unbounded_samples_ == 0 ? 0.0 : unbounded_ms_ / static_cast<double>(unbounded_samples_);
    }
    [[nodiscard]] double bounded_average_ms() const noexcept {
        return bounded_samples_ == 0 ? 0.0 : bounded_ms_ / static_cast<double>(bounded_samples_);
    }

private:
    std::uint32_t trial_samples_        = 16;
    double minimum_relative_gain_       = 0.005;
    std::uint32_t warmup_samples_       = 1;
    std::uint32_t unbounded_warmups_    = 0;
    std::uint32_t bounded_warmups_      = 0;
    bool bounded_                       = false;
    bool selected_                      = false;
    std::uint32_t unbounded_samples_    = 0;
    std::uint32_t bounded_samples_      = 0;
    double unbounded_ms_                = 0.0;
    double bounded_ms_                  = 0.0;
};

} // namespace ninfer
