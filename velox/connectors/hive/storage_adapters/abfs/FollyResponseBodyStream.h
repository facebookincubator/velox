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

#include <azure/core/context.hpp>
#include <azure/core/io/body_stream.hpp>

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>

namespace facebook::velox::filesystems {

/// Adapts an ABFS body transaction to the Azure SDK pull stream contract.
class FollyResponseBodyStream final : public Azure::Core::IO::BodyStream {
 public:
  /// Takes ownership of a body transaction and its optional decoded length.
  explicit FollyResponseBodyStream(
      std::unique_ptr<HttpBodyTransaction> transaction,
      std::optional<int64_t> length,
      std::chrono::milliseconds bodyIdleTimeout,
      std::shared_ptr<void> lifetime = nullptr);

  /// Abandons an unfinished body transaction.
  ~FollyResponseBodyStream() override;

  /// Returns the decoded body length, or -1 when it is unknown.
  int64_t Length() const override;

 private:
  size_t OnRead(
      uint8_t* buffer,
      size_t count,
      const Azure::Core::Context& context) override;

  std::unique_ptr<HttpBodyTransaction> transaction_;
  std::optional<int64_t> length_;
  std::chrono::milliseconds bodyIdleTimeout_;
  std::shared_ptr<void> lifetime_;
};

} // namespace facebook::velox::filesystems
