#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace ninfer {

// Chooses how many of a layer's ring runs go into one streamed-attention launch, using end-to-end
// decode round timings. Ported from the reference implementation's `ggml_cuda_kv_stream_span_tuner`
// and generalised from two candidates to a ladder.
//
// Why this has to be measured instead of derived: a narrower launch lets the consumer start as soon
// as the first run of pages lands, which is the only way to overlap the copies when the ring holds
// barely more than one layer's extent. It also turns one launch per layer into several, which costs
// fixed work per launch and leaves the device's waves ragged. Which effect wins depends on the card,
// the cache format and the ring's size, so the same binary measures it and keeps the winner.
//
// The ladder is ordered widest first - one launch per layer, then half the runs, then two, then one -
// so the *default* is the shape that always works, and every narrower step has to pay for itself.
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
        levels_.clear();
        samples_.clear();
        warmups_.clear();
        trial_          = 0;
        best_           = 0;
        best_ms_        = 0.0;
        chosen_         = 0;
        selected_       = false;
    }

    // The rounds-per-launch this tuner wants right now: the widest candidate asks for every run in
    // one launch, which the caller runs as its ordinary unframed path.
    [[nodiscard]] std::uint32_t candidate_runs_per_frame(std::uint32_t runs) noexcept {
        sync(runs);
        if (levels_.empty()) { return runs; }
        return levels_[selected_ ? best_ : trial_];
    }

    // True when the candidate under trial is narrower than the whole layer, i.e. the caller has to
    // frame the launch and wait run by run.
    [[nodiscard]] bool framed(std::uint32_t runs) noexcept {
        return candidate_runs_per_frame(runs) < std::max<std::uint32_t>(runs, 1U);
    }

    // One finished decode round: `elapsed_ms` of the whole round, whether it streamed at all (a round
    // with nothing to copy measures nothing about the pipeline), and how many runs its launches held.
    void observe(double elapsed_ms, bool streamed, std::uint32_t runs_per_frame,
                 std::uint32_t runs) noexcept {
        sync(runs);
        if (levels_.empty() || selected_ || !streamed || !std::isfinite(elapsed_ms) ||
            elapsed_ms <= 0.0) {
            return;
        }
        if (levels_[trial_] != runs_per_frame) { return; }
        std::uint32_t& warmups = warmups_[trial_];
        if (warmups < warmup_samples_) {
            ++warmups;
            return;
        }
        samples_[trial_].push_back(elapsed_ms);
        if (samples_[trial_].size() < trial_samples_) { return; }
        const double mean = average(trial_);
        // The widest candidate is the baseline every narrower one has to beat; a narrower one that
        // does not, or does not by the margin, leaves the winner where it was.
        if (best_ms_ == 0.0) {
            best_    = trial_;
            best_ms_ = mean;
        } else if (mean < best_ms_ * (1.0 - minimum_relative_gain_)) {
            best_    = trial_;
            best_ms_ = mean;
        }
        if (++trial_ >= levels_.size()) {
            trial_    = best_;
            chosen_   = levels_[best_];
            selected_ = true;
        }
    }

    [[nodiscard]] bool selected() const noexcept { return selected_; }
    [[nodiscard]] std::uint32_t chosen_runs_per_frame() const noexcept { return chosen_; }
    [[nodiscard]] double chosen_average_ms() const noexcept {
        return levels_.empty() ? 0.0 : average(best_);
    }
    [[nodiscard]] std::size_t trials() const noexcept { return levels_.size(); }

private:
    // The ladder for this layer width: every run in one launch, then half, then two, then one run.
    void sync(std::uint32_t runs) noexcept {
        if (runs == 0) { return; }
        if (!levels_.empty() && levels_.front() == runs) { return; }
        const std::uint32_t widest = runs;
        std::vector<std::uint32_t> ladder{widest};
        const std::uint32_t half = std::max<std::uint32_t>(1U, (runs + 1U) / 2U);
        for (const std::uint32_t step : {half, std::uint32_t{2}, std::uint32_t{1}}) {
            if (step < ladder.back() && step >= 1U) { ladder.push_back(step); }
        }
        reset();
        levels_  = std::move(ladder);
        samples_.assign(levels_.size(), {});
        warmups_.assign(levels_.size(), 0);
        best_ms_ = 0.0;
        // The widest candidate is the baseline: its mean is what the others must beat.
        samples_[0].reserve(trial_samples_);
        for (auto& list : samples_) { list.reserve(trial_samples_); }
        best_ = 0;
    }

    [[nodiscard]] double average(std::size_t index) const noexcept {
        if (index >= samples_.size() || samples_[index].empty()) { return 0.0; }
        double total = 0.0;
        for (const double value : samples_[index]) { total += value; }
        return total / static_cast<double>(samples_[index].size());
    }

    std::uint32_t trial_samples_        = 16;
    double minimum_relative_gain_       = 0.005;
    std::uint32_t warmup_samples_       = 1;
    std::vector<std::uint32_t> levels_;
    std::vector<std::vector<double>> samples_;
    std::vector<std::uint32_t> warmups_;
    std::size_t trial_    = 0;
    std::size_t best_     = 0;
    double best_ms_       = 0.0;
    std::uint32_t chosen_ = 0;
    bool selected_        = false;
};

} // namespace ninfer
