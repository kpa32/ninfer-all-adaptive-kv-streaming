// The streaming KV cache's exclusions (core/kv_stream_guards.h): every combination the planner rejects
// is rejected here, with the message the CLI prints, and every combination it serves is accepted.
//
// These are protected because each one is a way to corrupt a streamed request rather than to slow it
// down: a second sequence sharing the ring, a second device splitting every layer's KV, a checkpoint
// that reasons about whole-context device pages, or a pool sized by token arithmetic that cannot see
// the ring. The planner is the only caller, so a guard that loses its rejection has no other test.

#include "core/kv_stream_guards.h"

#include <iostream>
#include <map>
#include <string>

namespace {

using ninfer::KVStreamRejection;
using ninfer::KVStreamRequest;
using ninfer::kv_stream_rejection;
using ninfer::kv_stream_rejection_message;

std::map<std::string, int> failures;

void check(bool condition, const std::string& message) {
    if (!condition) { ++failures[message]; }
}

void check_rejection(KVStreamRejection actual, KVStreamRejection expected,
                     const std::string& message) {
    check(actual == expected, message);
}

// The request a delivered streaming run makes: one sequence, one device, measured pool, no retained
// context. Every case below changes exactly one field of it.
KVStreamRequest shipping_request() {
    KVStreamRequest request;
    request.enabled            = true;
    request.report             = true;
    request.capacity_automatic = true;
    request.sequences          = 1;
    request.devices            = 1;
    request.tensor_parallel    = false;
    request.retained_context   = false;
    return request;
}

void test_shipping_request_is_served() {
    const KVStreamRequest request = shipping_request();
    check_rejection(kv_stream_rejection(request), KVStreamRejection::None,
                    "a delivered streaming request is servable");
    check(std::string(kv_stream_rejection_message(KVStreamRejection::None)).empty(),
          "an accepted request has no message");
}

void test_reporting_without_streaming_is_rejected() {
    KVStreamRequest request = shipping_request();
    request.enabled         = false;
    check_rejection(kv_stream_rejection(request), KVStreamRejection::ReportWithoutStream,
                    "--kv-stream-report without --kv-stream is rejected");
    check(std::string(kv_stream_rejection_message(KVStreamRejection::ReportWithoutStream)) ==
              "--kv-stream-report measures --kv-stream: pass --kv-stream",
          "the report-only rejection names the flag to add");
}

void test_an_ordinary_request_is_untouched() {
    KVStreamRequest request;
    request.report = false;
    check_rejection(kv_stream_rejection(request), KVStreamRejection::None,
                    "a request without streaming is not constrained");
}

void test_an_explicit_capacity_is_rejected() {
    KVStreamRequest request = shipping_request();
    request.capacity_automatic = false;
    check_rejection(kv_stream_rejection(request), KVStreamRejection::CapacityNotAutomatic,
                    "an explicit capacity is rejected: the ring cannot be sized from tokens");
}

void test_a_second_sequence_is_rejected() {
    KVStreamRequest request = shipping_request();
    request.sequences      = 2;
    check_rejection(kv_stream_rejection(request), KVStreamRejection::ManySequences,
                    "a second sequence is rejected: the ring is claimed per layer per round");
}

void test_a_second_device_is_rejected() {
    KVStreamRequest piped = shipping_request();
    piped.devices         = 2;
    check_rejection(kv_stream_rejection(piped), KVStreamRejection::ManyDevices,
                    "pipeline stages are rejected: each stage splits the layers' KV");

    KVStreamRequest tensor = shipping_request();
    tensor.devices         = 2;
    tensor.tensor_parallel = true;
    check_rejection(kv_stream_rejection(tensor), KVStreamRejection::ManyDevices,
                    "tensor-parallel ranks are rejected: a layer's KV is divided over ranks");
}

void test_retained_context_is_rejected() {
    KVStreamRequest request = shipping_request();
    request.retained_context = true;
    check_rejection(kv_stream_rejection(request), KVStreamRejection::RetainedContext,
                    "a checkpoint or retained prefix is rejected");
}

void test_a_streamed_retention_is_served() {
    // A streamed run keeps one continuation in the arena the pinned host authority already is, so
    // retention is not the paged checkpoint the rejection above is about. Its endpoint state image
    // lives on the host shelf, one slot, and no device slot sits beside the window.
    KVStreamRequest request        = shipping_request();
    request.retained_context       = true;
    request.streamed_retention     = true;
    request.device_state_slots     = 0;
    request.host_state_slots       = 1;
    check_rejection(kv_stream_rejection(request), KVStreamRejection::None,
                    "a streamed retention is servable");

    request.device_state_slots = 1;
    check_rejection(kv_stream_rejection(request), KVStreamRejection::RetainedContextDeviceSlot,
                    "a device state slot is rejected beside the streamed window");

    request.device_state_slots = 0;
    request.host_state_slots   = 0;
    check_rejection(kv_stream_rejection(request), KVStreamRejection::RetainedContextNoHostSlot,
                    "a streamed retention without a host state slot has nowhere to keep its "
                    "endpoint");
}

void test_the_first_rejection_in_order_is_reported() {
    // A request that breaks several rules reports the one a user meets first: the report flag, then the
    // pool's sizing, then the sequence count, then the devices, then the retained context. That order is
    // what makes a single fix per attempt possible.
    KVStreamRequest request;
    request.enabled            = false;
    request.report             = true;
    request.capacity_automatic = false;
    request.sequences          = 4;
    request.devices            = 3;
    request.retained_context   = true;
    check_rejection(kv_stream_rejection(request), KVStreamRejection::ReportWithoutStream,
                    "the report flag is reported before the pool's sizing");

    request.enabled = true;
    check_rejection(kv_stream_rejection(request), KVStreamRejection::CapacityNotAutomatic,
                    "the pool's sizing is reported before the sequence count");

    request.capacity_automatic = true;
    check_rejection(kv_stream_rejection(request), KVStreamRejection::ManySequences,
                    "the sequence count is reported before the devices");

    request.sequences = 1;
    check_rejection(kv_stream_rejection(request), KVStreamRejection::ManyDevices,
                    "the devices are reported before the retained context");

    request.devices = 1;
    check_rejection(kv_stream_rejection(request), KVStreamRejection::RetainedContext,
                    "the retained context is reported last");
}

void test_every_rejection_has_a_message() {
    for (const KVStreamRejection rejection : {KVStreamRejection::CapacityNotAutomatic,
                                              KVStreamRejection::ManySequences,
                                              KVStreamRejection::ManyDevices,
                                              KVStreamRejection::RetainedContext,
                                              KVStreamRejection::ReportWithoutStream}) {
        check(std::string(kv_stream_rejection_message(rejection)).find("--kv-stream") !=
                  std::string::npos,
              "every rejection names a --kv-stream flag: " +
                  std::string(kv_stream_rejection_message(rejection)));
    }
}

} // namespace

int main() {
    test_shipping_request_is_served();
    test_reporting_without_streaming_is_rejected();
    test_an_ordinary_request_is_untouched();
    test_an_explicit_capacity_is_rejected();
    test_a_second_sequence_is_rejected();
    test_a_second_device_is_rejected();
    test_retained_context_is_rejected();
    test_a_streamed_retention_is_served();
    test_the_first_rejection_in_order_is_reported();
    test_every_rejection_has_a_message();

    if (failures.empty()) {
        std::cout << "kv stream guards: all checks passed\n";
        return 0;
    }
    for (const auto& [message, count] : failures) {
        std::cerr << "  failed (" << count << "x): " << message << '\n';
    }
    return 1;
}
