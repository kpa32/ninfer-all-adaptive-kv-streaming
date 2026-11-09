#include "core/kv_stream_feedback.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <string>

namespace {

using ninfer::KVStreamBusySample;
using ninfer::KVStreamDeadlineCounters;
using ninfer::kv_stream_copy_busy_ratio;
using ninfer::kv_stream_feedback_delta;
using ninfer::kv_stream_ring_peak_ratio;

std::map<std::string, int> failures;

void check(bool condition, const std::string& message) {
    if (!condition) { ++failures[message]; }
}

void check_near(double actual, double expected, double tolerance, const std::string& message) {
    check(std::fabs(actual - expected) <= tolerance, message);
}

// A window that advanced both counters reports the ratio of what it added, not of the totals:
// 145/17 over 120/12 is 5 misses in 25 samples, not 11.7%.
void the_delta_is_what_the_window_added() {
    const auto delta = kv_stream_feedback_delta({145, 17}, {120, 12});
    check(delta.valid, "a window that advanced is valid");
    check(delta.has_evaluation, "a window that advanced carries an evaluation");
    check(delta.samples == 25, "the window's samples are the difference");
    check(delta.misses == 5, "the window's misses are the difference");
    check_near(delta.miss_ratio, 0.20, 1e-12, "the ratio is the window's, not the totals'");
}

// No new samples means the controller has nothing to decide on: it must not read the previous
// window again, or it would count the same misses every round until it grows the ring for them.
void an_unchanged_window_has_no_evaluation() {
    const auto delta = kv_stream_feedback_delta({145, 17}, {145, 17});
    check(delta.valid, "an unchanged window is still valid");
    check(!delta.has_evaluation, "an unchanged window must not invent an evaluation");
    check_near(delta.miss_ratio, 0.0, 0.0, "an unmeasured window reports no misses");
}

// The counters live in pinned memory written by the device; a new run starts them at zero. That is
// not a measurement, and treating it as one would read "no misses since 145 samples" as idle.
void a_restarted_run_is_rejected() {
    const auto delta = kv_stream_feedback_delta({3, 1}, {145, 17});
    check(!delta.valid, "counters that moved backwards are rejected");
    check(delta.error != nullptr, "a rejected delta says why");
}

// More misses than samples cannot happen (each sample is one deadline observation), so a window
// claiming it is a broken sample, not a starved ring.
void impossible_counters_are_rejected() {
    const auto delta = kv_stream_feedback_delta({150, 30}, {145, 17});
    check(!delta.valid, "more misses than samples is rejected");
}

// The estimate scales one measured run to the window's uploads, then divides by the window's device
// time: 2 ms per run, 16 runs in a 40 ms window is 32/40 = 80% busy - the reference's saturation
// point.
void the_busy_estimate_scales_one_run_to_the_window() {
    KVStreamBusySample sample;
    sample.recorded        = true;
    sample.copy_seconds    = 0.002;
    sample.sampled_uploads = 1;
    sample.window_uploads  = 16;
    sample.window_seconds  = 0.040;
    check_near(kv_stream_copy_busy_ratio(sample), 0.80, 1e-12,
               "one 2 ms run of sixteen in 40 ms is 80% busy");
}

// A sampled run that already covers several uploads must not be multiplied by them again.
void a_batched_sample_is_not_counted_twice() {
    KVStreamBusySample sample;
    sample.recorded        = true;
    sample.copy_seconds    = 0.008;
    sample.sampled_uploads = 4;
    sample.window_uploads  = 8;
    sample.window_seconds  = 0.020;
    check_near(kv_stream_copy_busy_ratio(sample), 0.80, 1e-12,
               "a four-run sample stands for eight runs");
}

// An estimate above one would mean the link was busy for longer than the window existed.
void the_estimate_is_clamped() {
    KVStreamBusySample sample;
    sample.recorded        = true;
    sample.copy_seconds    = 0.100;
    sample.sampled_uploads = 1;
    sample.window_uploads  = 4;
    sample.window_seconds  = 0.010;
    check_near(kv_stream_copy_busy_ratio(sample), 1.0, 0.0, "the estimate is clamped to one");
}

// No recorded run, no window time, or no uploads are all "no estimate", never a busy of zero read
// as "the link is idle, shrink the ring".
void a_missing_sample_has_no_estimate() {
    KVStreamBusySample sample;
    sample.copy_seconds   = 0.002;
    sample.window_uploads = 4;
    sample.window_seconds = 0.010;
    check_near(kv_stream_copy_busy_ratio(sample), 0.0, 0.0, "an unrecorded sample has no estimate");

    sample.recorded       = true;
    sample.window_seconds = 0.0;
    check_near(kv_stream_copy_busy_ratio(sample), 0.0, 0.0, "a zero window has no estimate");

    sample.window_seconds = 0.010;
    sample.window_uploads = 0;
    check_near(kv_stream_copy_busy_ratio(sample), 0.0, 0.0, "a window without uploads has none");
}

// The controller reads occupancy as a fraction of the ring it planned, clamped: a ring that was
// repartitioned between the copy and the read must not report more than full.
void occupancy_is_a_fraction_of_the_planned_ring() {
    check_near(kv_stream_ring_peak_ratio(0, 100), 0.0, 0.0, "an unused ring is empty");
    check_near(kv_stream_ring_peak_ratio(30, 100), 0.30, 1e-12, "occupancy is a plain fraction");
    check_near(kv_stream_ring_peak_ratio(140, 100), 1.0, 0.0, "occupancy is clamped to one");
    check_near(kv_stream_ring_peak_ratio(5, 0), 0.0, 0.0, "a ring without slots reports nothing");
}

} // namespace

int main() {
    the_delta_is_what_the_window_added();
    an_unchanged_window_has_no_evaluation();
    a_restarted_run_is_rejected();
    impossible_counters_are_rejected();
    the_busy_estimate_scales_one_run_to_the_window();
    a_batched_sample_is_not_counted_twice();
    the_estimate_is_clamped();
    a_missing_sample_has_no_estimate();
    occupancy_is_a_fraction_of_the_planned_ring();
    if (!failures.empty()) {
        for (const auto& [message, count] : failures) {
            std::cerr << "FAIL (" << count << "x): " << message << '\n';
        }
        return 1;
    }
    std::cout << "kv stream feedback: all checks passed\n";
    return 0;
}
