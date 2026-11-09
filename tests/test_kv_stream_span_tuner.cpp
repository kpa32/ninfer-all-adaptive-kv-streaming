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

// A fresh tuner starts unbounded: the ordinary launch is the baseline, and the bounded pipeline has to
// earn its place.
void a_fresh_tuner_is_unbounded() {
    KVStreamSpanTuner tuner;
    check(!tuner.use_bounded(), "a fresh tuner runs the ordinary launch");
    check(!tuner.selected(), "a fresh tuner has selected nothing");
    check(tuner.collecting(), "a fresh tuner is collecting both pipelines");
}

// The reference's schedule: one warmup of each, then sixteen unbounded rounds, then sixteen bounded
// ones, and only then a decision.
void the_first_samples_are_unbounded_then_bounded() {
    KVStreamSpanTuner tuner(16, 0.005, 1);
    for (int round = 0; round < 40; ++round) {
        tuner.observe(100.0, true, tuner.use_bounded());
    }
    // 16 unbounded samples were collected first, so the bounded pipeline was measured afterwards.
    check(tuner.unbounded_average_ms() > 0.0, "the baseline was measured");
    check(tuner.bounded_average_ms() > 0.0, "the bounded pipeline was measured");
    check(tuner.selected(), "forty rounds select one of them");
    // Equal times cannot beat the half-percent margin.
    check(!tuner.use_bounded(), "a tie keeps the ordinary launch");
}

// The bounded pipeline wins only by more than the margin: a tenth of a percent is noise, not a result.
void the_margin_decides_close_calls() {
    KVStreamSpanTuner tuner(4, 0.005, 0);
    for (int round = 0; round < 4; ++round) { tuner.observe(100.0, true, false); }
    for (int round = 0; round < 4; ++round) { tuner.observe(99.8, true, true); }
    check(tuner.selected(), "the tuner decided");
    check(tuner.use_bounded(), "a two-tenths gain is past the half-percent margin");

    KVStreamSpanTuner narrow(4, 0.005, 0);
    for (int round = 0; round < 4; ++round) { narrow.observe(100.0, true, false); }
    for (int round = 0; round < 4; ++round) { narrow.observe(99.9, true, true); }
    check(narrow.selected(), "the narrow tuner decided too");
    check(!narrow.use_bounded(), "a one-tenth gain is inside the margin");
}

// A round that streamed nothing says nothing about the pipeline, and a non-positive time is not a
// measurement: neither may enter either average.
void rounds_that_measure_nothing_are_ignored() {
    KVStreamSpanTuner tuner(2, 0.005, 0);
    for (int round = 0; round < 20; ++round) {
        tuner.observe(50.0, false, false);
        tuner.observe(0.0, true, false);
        tuner.observe(-1.0, true, true);
    }
    check(!tuner.selected(), "rounds without streaming never select a pipeline");
    check(tuner.unbounded_average_ms() == 0.0, "an unstreamed round contributes no time");
    check(tuner.bounded_average_ms() == 0.0, "a non-positive time contributes none either");
    check(tuner.collecting(), "the tuner keeps waiting for real samples");
}

// The decision is final until the layout changes: a later slow round must not flip it, or the decode
// loop would alternate between pipelines on noise.
void a_selected_choice_is_final_until_reset() {
    KVStreamSpanTuner tuner(2, 0.005, 0);
    for (int round = 0; round < 2; ++round) { tuner.observe(100.0, true, false); }
    for (int round = 0; round < 2; ++round) { tuner.observe(80.0, true, true); }
    check(tuner.use_bounded(), "the bounded pipeline won");

    for (int round = 0; round < 20; ++round) { tuner.observe(500.0, true, false); }
    check(tuner.use_bounded(), "a selected choice is not revisited");

    tuner.reset();
    check(!tuner.use_bounded(), "a reset returns to the baseline");
    check(!tuner.selected(), "a reset forgets the decision");
    check(tuner.unbounded_average_ms() == 0.0 && tuner.bounded_average_ms() == 0.0,
          "a reset forgets the averages");
}

// While collecting, the caller alternates with the tuner: it must ask for a bounded round once the
// baseline is complete, or the second pipeline is never measured.
void the_caller_is_told_which_pipeline_to_run() {
    KVStreamSpanTuner tuner(2, 0.005, 0);
    check(!tuner.use_bounded(), "the baseline runs first");
    tuner.observe(100.0, true, tuner.use_bounded());
    check(!tuner.use_bounded(), "the baseline is not complete after one sample");
    tuner.observe(100.0, true, tuner.use_bounded());
    check(tuner.use_bounded(), "the bounded pipeline runs once the baseline is complete");
    check(tuner.collecting(), "the tuner is still collecting");
}

} // namespace

int main() {
    a_fresh_tuner_is_unbounded();
    the_first_samples_are_unbounded_then_bounded();
    the_margin_decides_close_calls();
    rounds_that_measure_nothing_are_ignored();
    a_selected_choice_is_final_until_reset();
    the_caller_is_told_which_pipeline_to_run();
    if (!failures.empty()) {
        for (const auto& [message, count] : failures) {
            std::cerr << "FAIL (" << count << "x): " << message << '\n';
        }
        return 1;
    }
    std::cout << "kv stream span tuner: all checks passed\n";
    return 0;
}
