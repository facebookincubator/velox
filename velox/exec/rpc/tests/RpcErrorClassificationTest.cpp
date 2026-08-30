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

#include <stdexcept>

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

} // namespace
} // namespace facebook::velox::exec::rpc
