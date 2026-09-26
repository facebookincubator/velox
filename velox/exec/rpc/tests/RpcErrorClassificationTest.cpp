/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/// Unit tests for isTimeout() — the helper that recognizes timeout failures so
/// the transports can tag them RPCErrorKind::kTimeout (a hard-overload signal
/// for the congestion policy) instead of the generic kBackendError.

#include "velox/exec/rpc/RpcErrorClassification.h"
#include "velox/expression/rpc/AsyncRPCFunction.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/rpc/RPCTypes.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <folly/ExceptionWrapper.h>
#include <folly/futures/Future.h>
#include <gtest/gtest.h>

#include <thrift/lib/cpp/TApplicationException.h>
#include <thrift/lib/cpp/transport/TTransportException.h>

namespace facebook::velox::exec::rpc {
namespace {

using apache::thrift::TApplicationException;
using apache::thrift::transport::TTransportException;

// Client-side processing/transport timeout (what setProcessingTimeoutMs fires).
TEST(RpcErrorClassificationTest, transportTimedOutIsTimeout) {
  auto ew = folly::make_exception_wrapper<TTransportException>(
      TTransportException::TIMED_OUT, "processing timeout");
  EXPECT_TRUE(isTimeout(ew));
}

// A non-timeout transport error must NOT be classified as a timeout.
TEST(RpcErrorClassificationTest, transportNonTimeoutIsNotTimeout) {
  auto ew = folly::make_exception_wrapper<TTransportException>(
      TTransportException::END_OF_FILE, "eof");
  EXPECT_FALSE(isTimeout(ew));
}

// Server-reported application timeout.
TEST(RpcErrorClassificationTest, applicationTimeoutIsTimeout) {
  auto ew = folly::make_exception_wrapper<TApplicationException>(
      TApplicationException::TIMEOUT, "server timeout");
  EXPECT_TRUE(isTimeout(ew));
}

// A non-timeout application error must NOT be classified as a timeout.
TEST(RpcErrorClassificationTest, applicationNonTimeoutIsNotTimeout) {
  auto ew = folly::make_exception_wrapper<TApplicationException>(
      TApplicationException::INTERNAL_ERROR, "boom");
  EXPECT_FALSE(isTimeout(ew));
}

// folly .within(deadline) expiry.
TEST(RpcErrorClassificationTest, follyFutureTimeoutIsTimeout) {
  auto ew = folly::make_exception_wrapper<folly::FutureTimeout>();
  EXPECT_TRUE(isTimeout(ew));
}

// A generic error is not a timeout (must fall through to kBackendError).
TEST(RpcErrorClassificationTest, genericErrorIsNotTimeout) {
  auto ew = folly::make_exception_wrapper<std::runtime_error>("generic");
  EXPECT_FALSE(isTimeout(ew));
}

// A transport that already classified must win: re-deriving from the
// exception type would turn a rate limit into a plain backend error and cost
// the congestion policy its immediate backoff.
TEST(RpcErrorClassificationTest, classifiedErrorKeepsItsKind) {
  auto ew = folly::make_exception_wrapper<RpcClassifiedError>(
      velox::rpc::RPCErrorKind::kRateLimited, "429 from backend");
  EXPECT_EQ(errorKindFor(ew), velox::rpc::RPCErrorKind::kRateLimited);
}

// Including for a kind no exception type could reveal on its own.
TEST(RpcErrorClassificationTest, classifiedTimeoutSurvivesAPlainRuntimeError) {
  auto ew = folly::make_exception_wrapper<RpcClassifiedError>(
      velox::rpc::RPCErrorKind::kTimeout, "polling timeout after 30 attempts");
  EXPECT_EQ(errorKindFor(ew), velox::rpc::RPCErrorKind::kTimeout);
  // The same message as a bare runtime_error is invisible to isTimeout(),
  // which is exactly why the kind has to be carried.
  auto bare = folly::make_exception_wrapper<std::runtime_error>(
      "polling timeout after 30 attempts");
  EXPECT_EQ(errorKindFor(bare), velox::rpc::RPCErrorKind::kBackendError);
}

// An unclassified fault of ours still resolves to internal.
TEST(RpcErrorClassificationTest, unclassifiedInternalStillResolves) {
  auto ew = folly::make_exception_wrapper<std::bad_alloc>();
  EXPECT_EQ(errorKindFor(ew), velox::rpc::RPCErrorKind::kInternalError);
}

// ── Inline payload storage ──────────────────────────────────────────────────

namespace {
// Counts its own lifetime so a test can prove the payload destroys exactly
// once no matter how many times it is moved.
struct LifetimeProbe {
  static int liveCount;
  int value{0};

  explicit LifetimeProbe(int v) : value(v) {
    ++liveCount;
  }
  LifetimeProbe(LifetimeProbe&& other) noexcept : value(other.value) {
    ++liveCount;
  }
  LifetimeProbe(const LifetimeProbe&) = delete;
  LifetimeProbe& operator=(const LifetimeProbe&) = delete;
  LifetimeProbe& operator=(LifetimeProbe&&) = delete;
  ~LifetimeProbe() {
    --liveCount;
  }

  int64_t retainedBytes() const noexcept {
    return 37;
  }
};
int LifetimeProbe::liveCount = 0;

struct StringPayloadProbe {
  std::string value;

  int64_t retainedBytes() const noexcept {
    return static_cast<int64_t>(value.capacity());
  }
};

struct VectorPayloadProbe {
  std::vector<float> values;

  int64_t retainedBytes() const noexcept {
    return static_cast<int64_t>(values.capacity() * sizeof(float));
  }
};

struct FixedSizePayloadProbe {
  int64_t bytes;

  int64_t retainedBytes() const noexcept {
    return bytes;
  }
};

struct LegacyPayloadProbe {
  int64_t value;
};

struct WrongTypeRetainedBytesPayload {
  [[maybe_unused]] size_t retainedBytes() const noexcept;
};

struct ThrowingRetainedBytesPayload {
  [[maybe_unused]] int64_t retainedBytes() const;
};

struct MutableRetainedBytesPayload {
  [[maybe_unused]] int64_t retainedBytes() noexcept;
};
} // namespace

// Reading a payload as the wrong type is caught, not reinterpreted. This is
// the guarantee the old dynamic_cast provided and the reason the stored type
// is compared rather than assumed.
TEST(RpcErrorClassificationTest, payloadReadAsTheWrongTypeThrows) {
  velox::rpc::RPCResponse response;
  response.setPayload(StringPayloadProbe{"some text"});

  EXPECT_TRUE(response.payload().holds<StringPayloadProbe>());
  EXPECT_FALSE(response.payload().holds<VectorPayloadProbe>());
  VELOX_ASSERT_THROW(
      response.payload().get<VectorPayloadProbe>(), "is not of the type");
}

// A response is handed along a chain of continuations, so the payload has to
// survive being moved and must not be destroyed twice on the way.
TEST(RpcErrorClassificationTest, payloadSurvivesMovesAndDestroysExactlyOnce) {
  ASSERT_EQ(LifetimeProbe::liveCount, 0);
  {
    velox::rpc::RPCResponse first;
    first.setPayload(LifetimeProbe{7});
    EXPECT_EQ(first.payload().get<LifetimeProbe>().value, 7);

    // Across a move, as a completion handler would.
    auto second = std::move(first);
    EXPECT_EQ(second.payload().get<LifetimeProbe>().value, 7);

    // And into a container, as the scatter does.
    std::vector<velox::rpc::RPCResponse> batch;
    batch.push_back(std::move(second));
    batch.emplace_back();
    EXPECT_EQ(batch[0].payload().get<LifetimeProbe>().value, 7);
    EXPECT_EQ(LifetimeProbe::liveCount, 1);
  }
  EXPECT_EQ(LifetimeProbe::liveCount, 0) << "payload leaked or double-freed";
}

TEST(RpcErrorClassificationTest, payloadReportsRetainedBytesAfterMove) {
  velox::rpc::RPCResponse response;
  response.setPayload(LifetimeProbe{7});
  EXPECT_EQ(response.retainedBytes(), 37);

  auto moved = std::move(response);
  // RpcPayload's move contract leaves the source empty; inspecting that
  // documented moved-from state is the behavior under test.
  // NOLINTNEXTLINE(bugprone-use-after-move)
  EXPECT_EQ(response.retainedBytes(), 0);
  EXPECT_EQ(moved.retainedBytes(), 37);
}

TEST(RpcErrorClassificationTest, textPayloadReportsReservedCapacity) {
  std::string text;
  text.reserve(4096);
  text.resize(17, 'x');
  const auto capacity = text.capacity();

  velox::rpc::RPCResponse response;
  response.setPayload(makeTextPayload(std::move(text)));
  EXPECT_EQ(response.retainedBytes(), capacity);
}

TEST(RpcErrorClassificationTest, errorReportsDiagnosticBytes) {
  std::string message;
  message.reserve(4096);
  message.resize(101, 'e');
  const auto capacity = message.capacity();
  velox::rpc::RPCResponse response;
  response.setError(
      velox::rpc::RPCErrorKind::kBackendError, std::move(message));
  EXPECT_EQ(response.retainedBytes(), capacity);
}

TEST(RpcErrorClassificationTest, legacyPayloadWithoutSizeReportsZero) {
  velox::rpc::RPCResponse response;
  response.setPayload(LegacyPayloadProbe{7});
  EXPECT_EQ(response.retainedBytes(), 0);
}

TEST(RpcErrorClassificationTest, malformedSizeReportersAreRejected) {
  EXPECT_FALSE((std::is_constructible_v<
                velox::rpc::RpcPayload,
                WrongTypeRetainedBytesPayload>));
  EXPECT_FALSE((std::is_constructible_v<
                velox::rpc::RpcPayload,
                ThrowingRetainedBytesPayload>));
  EXPECT_FALSE((std::is_constructible_v<
                velox::rpc::RpcPayload,
                MutableRetainedBytesPayload>));
}

TEST(RpcErrorClassificationTest, retainedByteTotalRejectsInvalidSizes) {
  std::vector<velox::rpc::RPCResponse> responses(1);
  responses[0].setPayload(FixedSizePayloadProbe{-1});
  VELOX_ASSERT_THROW(
      velox::rpc::totalResponseRetainedBytes(responses), "non-negative");

  responses.resize(2);
  responses[0].setPayload(
      FixedSizePayloadProbe{std::numeric_limits<int64_t>::max()});
  responses[1].setPayload(FixedSizePayloadProbe{1});
  VELOX_ASSERT_THROW(
      velox::rpc::totalResponseRetainedBytes(responses), "overflow");
}

// The point of storing inline: a successful response costs no allocation of
// its own. Proven by buffer identity rather than by timing -- a prebuilt
// string's heap buffer must still be at the same address after it is stored,
// which it cannot be if the wrapper allocated and copied.
TEST(RpcErrorClassificationTest, storingAPayloadReusesTheCallersBuffer) {
  std::string text("a completion long enough to need its own heap buffer");
  const auto* before = text.data();

  velox::rpc::RPCResponse response;
  response.setPayload(StringPayloadProbe{std::move(text)});

  EXPECT_EQ(response.payload().get<StringPayloadProbe>().value.data(), before)
      << "the payload was copied into fresh storage rather than moved in";

  // And again across the moves a response makes on its way to buildOutput().
  auto moved = std::move(response);
  std::vector<velox::rpc::RPCResponse> batch;
  batch.push_back(std::move(moved));
  EXPECT_EQ(batch[0].payload().get<StringPayloadProbe>().value.data(), before)
      << "a move of the response reallocated the payload's buffer";
}

// The same for the other payload that actually flows.
TEST(RpcErrorClassificationTest, storingAVectorPayloadReusesItsBuffer) {
  std::vector<float> values{1.0f, 2.0f, 3.0f};
  const auto* before = values.data();

  velox::rpc::RPCResponse response;
  response.setPayload(VectorPayloadProbe{std::move(values)});

  EXPECT_EQ(response.payload().get<VectorPayloadProbe>().values.data(), before);
}

// Storage is inline, so the response owns no pointer to separate payload
// memory: the whole value lives in the response's own bytes.
TEST(RpcErrorClassificationTest, payloadStorageIsInline) {
  static_assert(
      sizeof(std::string) <= velox::rpc::RpcPayload::kMaxSize,
      "a text payload must fit inline");
  static_assert(
      sizeof(std::vector<float>) <= velox::rpc::RpcPayload::kMaxSize,
      "an embedding payload must fit inline");
}

// ── The unset response ──────────────────────────────────────────────────────

// A producer that returns without setting an outcome reads as an error, so
// "neither payload nor error" is not a representable state. It is tagged
// kUnset rather than kBackendError; the operator rejects this construction
// sentinel if it reaches output.
TEST(RpcErrorClassificationTest, unsetResponseReadsAsUnsetError) {
  velox::rpc::RPCResponse response;
  response.rowId = 7;
  EXPECT_TRUE(response.hasError());
  EXPECT_EQ(response.errorKind(), velox::rpc::RPCErrorKind::kUnset);
}

// kNone means "not an error". Storing it as one makes hasError() and
// errorKind() disagree, and the metric switches then drop the row without
// counting it anywhere -- so the state is rejected rather than documented.
TEST(RpcErrorClassificationTest, setErrorRejectsKNone) {
  velox::rpc::RPCResponse response;
  VELOX_ASSERT_THROW(
      response.setError(velox::rpc::RPCErrorKind::kNone, "no cause"),
      "must name a cause");
}

// Every other kind is accepted, so the guard is specific rather than a blanket
// restriction on setError().
TEST(RpcErrorClassificationTest, setErrorAcceptsANamedCause) {
  velox::rpc::RPCResponse response;
  response.setError(velox::rpc::RPCErrorKind::kBackendError, "refused");
  EXPECT_EQ(response.errorKind(), velox::rpc::RPCErrorKind::kBackendError);
}

// Once filled, the same accessors are ordinary reads.
TEST(RpcErrorClassificationTest, filledResponseReadsBack) {
  auto response = velox::rpc::RPCResponse::failed(
      3, velox::rpc::RPCErrorKind::kTimeout, "deadline");
  EXPECT_EQ(response.errorKind(), velox::rpc::RPCErrorKind::kTimeout);
  EXPECT_EQ(response.error().message, "deadline");
}

} // namespace
} // namespace facebook::velox::exec::rpc
