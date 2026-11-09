#include "core/kv_stream_feedback.h"

#include <algorithm>

namespace ninfer {

KVStreamFeedbackDelta kv_stream_feedback_delta(KVStreamDeadlineCounters current,
                                              KVStreamDeadlineCounters previous) noexcept {
    KVStreamFeedbackDelta result;
    if (current.samples < previous.samples || current.misses < previous.misses) {
        result.error = "KV stream deadline counters moved backwards";
        return result;
    }

    result.samples = current.samples - previous.samples;
    result.misses  = current.misses - previous.misses;
    if (result.misses > result.samples) {
        result.error = "KV stream deadline misses exceed samples";
        return result;
    }

    result.valid          = true;
    result.has_evaluation = result.samples != 0;
    if (result.has_evaluation) {
        result.miss_ratio = static_cast<double>(result.misses) /
                            static_cast<double>(result.samples);
    }
    return result;
}

double kv_stream_copy_busy_ratio(const KVStreamBusySample& sample) noexcept {
    if (!sample.recorded || sample.sampled_uploads == 0 || sample.window_uploads == 0 ||
        sample.copy_seconds <= 0.0 || sample.window_seconds <= 0.0) {
        return 0.0;
    }
    const double scaled = sample.copy_seconds *
                          static_cast<double>(sample.window_uploads) /
                          (static_cast<double>(sample.sampled_uploads) * sample.window_seconds);
    return std::min(1.0, scaled);
}

double kv_stream_ring_peak_ratio(std::uint32_t peak_slots, std::uint32_t ring_slots) noexcept {
    if (ring_slots == 0) { return 0.0; }
    return std::min(1.0, static_cast<double>(peak_slots) / static_cast<double>(ring_slots));
}

} // namespace ninfer
