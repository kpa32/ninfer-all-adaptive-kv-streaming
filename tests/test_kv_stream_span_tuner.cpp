#include "core/kv_stream_span_tuner.h"

#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <algorithm>
#include <vector>

namespace {

using ninfer::KVStreamSpanTuner;

std::map<std::string, int> failures;

void check(bool condition, const std::string& message) {
    if (!condition) { ++failures[message]; }
}

// The ladder for a layer of `runs` runs: every run in one launch, then half, then two, then one.
std::vector<std::uint32_t> ladder(std::uint32_t runs) {
    std::vector<std::uint32_t> steps{runs};
    for (const std::uint32_t step : {std::max<std::uint32_t>(1U, (runs + 1U) / 2U), std::uint32_t{2},
                                     std::uint32_t{1}}) {
        if (step < steps.back()) { steps.push_back(step); }
    }
    return steps;
}

// Runs the whole ladder, giving each step `ms` as its round time, and checks that the tuner asked for
// exactly that step while it was measured.
void drive(KVStreamSpanTuner& tuner, std::uint32_t runs, const std::vector<double>& ms,
           std::uint32_t samples) {
    const std::vector<std::uint32_t> steps = ladder(runs);
    for (std::size_t level = 0; level < steps.size(); ++level) {
        for (std::uint32_t sample = 0; sample < samples; ++sample) {
            const std::uint32_t asked = tuner.candidate_runs_per_frame(runs);
            check(asked == steps[level], "the tuner measures the step it is about to report");
            tuner.observe(ms[level], true, asked, runs);
        }
    }
}

// A fresh tuner asks for the widest launch: the ordinary unframed path is the baseline, and every
// narrower step has to earn its place by making the round faster.
void a_fresh_tuner_runs_unframed() {
    KVStreamSpanTuner tuner;
    check(tuner.candidate_runs_per_frame(8) == 8, "a fresh tuner keeps every run in one launch");
    check(!tuner.framed(8), "a fresh tuner frames nothing");
    check(!tuner.selected(), "a fresh tuner has selected nothing");
}

// The ladder is measured in full - four steps for eight runs - and only then is the fastest kept.
void the_ladder_is_measured_in_full() {
    KVStreamSpanTuner tuner(2, 0.005, 0);
    drive(tuner, 8, {100.0, 110.0, 120.0, 130.0}, 2);
    check(tuner.selected(), "the whole ladder has been measured");
    check(tuner.trials() == 4, "the ladder is four steps wide for eight runs");
    check(tuner.candidate_runs_per_frame(8) == 8, "the fastest step wins: the baseline was fastest");
}

// A narrower step wins only by more than the margin: a tenth of a percent is noise, not a result.
void the_margin_decides_close_calls() {
    KVStreamSpanTuner tuner(2, 0.005, 0);
    drive(tuner, 8, {100.0, 99.0, 130.0, 140.0}, 2);
    check(tuner.candidate_runs_per_frame(8) == 4, "a one-percent gain takes the step");

    KVStreamSpanTuner narrow(2, 0.005, 0);
    drive(narrow, 8, {100.0, 99.9, 130.0, 140.0}, 2);
    check(narrow.candidate_runs_per_frame(8) == 8, "a one-tenth gain is inside the margin");
}

// A round that streamed nothing, a non-positive time, and a round whose shape is not the candidate
// under trial are not measurements of it.
void rounds_that_measure_nothing_are_ignored() {
    KVStreamSpanTuner tuner(2, 0.005, 0);
    for (int round = 0; round < 20; ++round) {
        tuner.observe(50.0, false, 8, 8);
        tuner.observe(0.0, true, 8, 8);
        tuner.observe(-1.0, true, 8, 8);
        tuner.observe(40.0, true, 1, 8);
    }
    check(!tuner.selected(), "rounds that measure nothing never select a step");
    check(tuner.chosen_average_ms() == 0.0, "and they contribute no time");
    check(tuner.candidate_runs_per_frame(8) == 8, "the baseline is still the candidate");

    tuner.observe(90.0, true, 8, 8);
    check(tuner.candidate_runs_per_frame(8) == 8, "a single real sample is still one sample");
}

// The decision is final until the layout changes: a later slow round must not flip it, or the decode
// loop would alternate between shapes on noise.
void a_selected_choice_is_final_until_reset() {
    KVStreamSpanTuner tuner(2, 0.005, 0);
    drive(tuner, 8, {100.0, 70.0, 90.0, 120.0}, 2);
    check(tuner.selected(), "the ladder finished");
    check(tuner.candidate_runs_per_frame(8) == 4, "the three-percent step that won stays chosen");

    std::vector<double> noise;
    for (int round = 0; round < 40; ++round) { tuner.observe(500.0, true, 1, 8); }
    check(tuner.candidate_runs_per_frame(8) == 4, "a selected choice is not revisited");

    tuner.reset();
    check(tuner.candidate_runs_per_frame(8) == 8, "a reset returns to the baseline");
    check(!tuner.selected(), "a reset forgets the decision");
}

// A layer whose extent changed is a different trade: the ladder is rebuilt for the new run count.
void a_new_extent_rebuilds_the_ladder() {
    KVStreamSpanTuner tuner(2, 0.005, 0);
    drive(tuner, 8, {100.0, 70.0, 90.0, 120.0}, 2);
    check(tuner.candidate_runs_per_frame(8) == 4, "the narrower step won at eight runs");
    check(tuner.candidate_runs_per_frame(5) == 5,
          "a different extent starts from its own baseline again");
    check(!tuner.selected(), "and its ladder is not resolved yet");
}

} // namespace

int main() {
    a_fresh_tuner_runs_unframed();
    the_ladder_is_measured_in_full();
    the_margin_decides_close_calls();
    rounds_that_measure_nothing_are_ignored();
    a_selected_choice_is_final_until_reset();
    a_new_extent_rebuilds_the_ladder();
    if (!failures.empty()) {
        for (const auto& [message, count] : failures) {
            std::cerr << "FAIL (" << count << "x): " << message << '\n';
        }
        return 1;
    }
    std::cout << "kv stream span tuner: all checks passed\n";
    return 0;
}
