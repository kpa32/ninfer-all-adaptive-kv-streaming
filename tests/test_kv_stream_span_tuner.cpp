#include "core/kv_stream_span_tuner.h"

#include <cstdint>
#include <iostream>
#include <map>
#include <string>

namespace {

using ninfer::KVStreamSpanTuner;

std::map<std::string, int> failures;

void check(bool condition, const std::string& message) {
    if (!condition) { ++failures[message]; }
}

// A fresh tuner asks for the widest launch: the ordinary unframed path is the baseline, and every
// narrower step has to earn its place by making the round faster.
void a_fresh_tuner_runs_unframed() {
    KVStreamSpanTuner tuner;
    check(tuner.candidate_runs_per_frame(8) == 8, "a fresh tuner keeps every run in one launch");
    check(!tuner.framed(8), "a fresh tuner frames nothing");
    check(!tuner.selected(), "a fresh tuner has selected nothing");
}

// The ladder narrows only after the baseline is complete, and it reports the candidate it is
// measuring so the caller runs the shape being timed.
void the_ladder_narrows_after_the_baseline() {
    KVStreamSpanTuner tuner(2, 0.005, 0);
    check(tuner.candidate_runs_per_frame(8) == 8, "the baseline runs first");
    tuner.observe(100.0, true, 8, 8);
    check(tuner.candidate_runs_per_frame(8) == 8, "one sample is not a trial");
    tuner.observe(100.0, true, 8, 8);
    check(tuner.candidate_runs_per_frame(8) == 4, "the ladder narrows to half the runs");
    tuner.observe(110.0, true, 4, 8);
    tuner.observe(110.0, true, 4, 8);
    check(tuner.candidate_runs_per_frame(8) == 2, "then to two runs");
    tuner.observe(120.0, true, 2, 8);
    tuner.observe(120.0, true, 2, 8);
    check(tuner.candidate_runs_per_frame(8) == 1, "then to one run");
    tuner.observe(130.0, true, 1, 8);
    tuner.observe(130.0, true, 1, 8);
    check(tuner.selected(), "the whole ladder has been measured");
    check(tuner.candidate_runs_per_frame(8) == 8, "the fastest step wins, and the baseline was fastest");
    check(tuner.trials() == 4, "the ladder is four steps wide for eight runs");
}

// A narrower step wins only by more than the margin: a tenth of a percent is noise, not a result.
void the_margin_decides_close_calls() {
    KVStreamSpanTuner tuner(2, 0.005, 0);
    for (int round = 0; round < 2; ++round) { tuner.observe(100.0, true, 8, 8); }
    for (int round = 0; round < 2; ++round) { tuner.observe(99.0, true, 4, 8); }
    check(tuner.candidate_runs_per_frame(8) == 4, "a one-percent gain takes the step");

    KVStreamSpanTuner narrow(2, 0.005, 0);
    for (int round = 0; round < 2; ++round) { narrow.observe(100.0, true, 8, 8); }
    for (int round = 0; round < 2; ++round) { narrow.observe(99.9, true, 4, 8); }
    check(narrow.candidate_runs_per_frame(8) == 4,
          "a one-tenth gain is inside the margin, so the baseline still leads the next step");
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

    // One zero-length sample is enough either way: the trial completes on real samples only.
    tuner.observe(90.0, true, 8, 8);
    check(tuner.candidate_runs_per_frame(8) == 8, "a single real sample is still one sample");
}

// The decision is final until the layout changes: a later slow round must not flip it, or the decode
// loop would alternate between shapes on noise.
void a_selected_choice_is_final_until_reset() {
    KVStreamSpanTuner tuner(2, 0.005, 0);
    for (int round = 0; round < 2; ++round) { tuner.observe(100.0, true, 8, 8); }
    for (int round = 0; round < 2; ++round) { tuner.observe(80.0, true, 4, 8); }
    tuner.observe(80.0, true, 2, 8);
    tuner.observe(80.0, true, 1, 8);
    for (int round = 0; round < 40; ++round) {
        for (int sample = 0; sample < 2; ++sample) { tuner.observe(500.0, true, 2, 8); }
    }
    check(tuner.selected(), "the ladder finished");
    check(tuner.candidate_runs_per_frame(8) == 4, "the narrow step that won stays chosen");

    tuner.reset();
    check(tuner.candidate_runs_per_frame(8) == 8, "a reset returns to the baseline");
    check(!tuner.selected(), "a reset forgets the decision");
}

// A layer whose extent changed is a different trade: the ladder is rebuilt for the new run count.
void a_new_extent_rebuilds_the_ladder() {
    KVStreamSpanTuner tuner(2, 0.005, 0);
    for (int round = 0; round < 2; ++round) { tuner.observe(100.0, true, 8, 8); }
    for (int round = 0; round < 2; ++round) { tuner.observe(70.0, true, 4, 8); }
    check(tuner.candidate_runs_per_frame(8) == 4, "the narrower step won at eight runs");
    check(tuner.candidate_runs_per_frame(5) == 5,
          "a different extent starts from its own baseline again");
    check(!tuner.selected(), "and its ladder is not resolved yet");
}

} // namespace

int main() {
    a_fresh_tuner_runs_unframed();
    the_ladder_narrows_after_the_baseline();
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
