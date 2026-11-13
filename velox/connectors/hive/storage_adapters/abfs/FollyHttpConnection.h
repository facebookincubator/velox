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

#pragma once

#include "velox/connectors/hive/storage_adapters/abfs/HttpConnection.h"

#include <folly/io/async/AsyncTransport.h>
#include <folly/io/async/EventBase.h>

#include <memory>

namespace facebook::velox::filesystems {

/// Implements one persistent HTTP/1.1 transaction over a Folly transport.
class FollyHttpConnection final : public HttpConnection {
 public:
  /// Takes ownership of a connected transport whose EventBase remains affine.
  explicit FollyHttpConnection(
      folly::AsyncTransportWrapper::UniquePtr transport);

  /// Closes the transport and abandons an active response.
  ~FollyHttpConnection() override;

  /// Sends one request from the transport's EventBase fiber.
  HttpResponseTransaction send(
      const HttpRequest& request,
      const HttpLimits& limits,
      const HttpTimeouts& timeouts,
      HttpTransactionRelease release) override;

  /// Returns the EventBase to which this connection is permanently bound.
  folly::EventBase* eventBase() const noexcept;

  /// Reports whether the transport and its released transaction can be reused.
  bool usable() const noexcept;

  class TransportHolder;
  class TransactionState;

 private:
  std::shared_ptr<TransportHolder> transport_;
  std::shared_ptr<TransactionState> activeTransaction_;
};

} // namespace facebook::velox::filesystems
