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

#include "velox/connectors/hive/storage_adapters/abfs/FollyResponseBodyStream.h"

#include <azure/core/http/http.hpp>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace facebook::velox::filesystems {
namespace {

constexpr char kBodyStreamErrorPrefix[] = "ABFS Azure response body failure: ";

Azure::Core::Http::TransportException transportException(
    const std::exception& exception) {
  return Azure::Core::Http::TransportException(
      std::string(kBodyStreamErrorPrefix) + exception.what());
}

} // namespace

FollyResponseBodyStream::FollyResponseBodyStream(
    std::unique_ptr<HttpBodyTransaction> transaction,
    std::optional<int64_t> length,
    std::chrono::milliseconds bodyIdleTimeout,
    std::shared_ptr<void> lifetime)
    : transaction_(std::move(transaction)),
      length_(length),
      bodyIdleTimeout_(bodyIdleTimeout),
      lifetime_(std::move(lifetime)) {
  if (transaction_ == nullptr) {
    throw std::invalid_argument(
        "ABFS Azure response body requires a transaction");
  }
  if (length_.has_value() && *length_ < 0) {
    throw std::invalid_argument(
        "ABFS Azure response body length must be non-negative");
  }
  if (bodyIdleTimeout_.count() <= 0) {
    throw std::invalid_argument(
        "ABFS Azure response body timeout must be positive");
  }
}

FollyResponseBodyStream::~FollyResponseBodyStream() {
  if (transaction_ != nullptr && !transaction_->complete()) {
    transaction_->abandon();
  }
}

int64_t FollyResponseBodyStream::Length() const {
  return length_.value_or(-1);
}

size_t FollyResponseBodyStream::OnRead(
    uint8_t* buffer,
    size_t count,
    const Azure::Core::Context& context) {
  if (count == 0) {
    return 0;
  }
  if (buffer == nullptr) {
    throw Azure::Core::Http::TransportException(
        std::string(kBodyStreamErrorPrefix) + "null read buffer");
  }
  try {
    context.ThrowIfCancelled();
    const auto bytes = transaction_->read(buffer, count, bodyIdleTimeout_);
    if (bytes == 0 && !transaction_->complete()) {
      throw std::runtime_error("body returned zero bytes before completion");
    }
    return bytes;
  } catch (const Azure::Core::Http::TransportException&) {
    transaction_->abandon();
    throw;
  } catch (const std::exception& exception) {
    transaction_->abandon();
    throw transportException(exception);
  } catch (...) {
    transaction_->abandon();
    throw Azure::Core::Http::TransportException(
        std::string(kBodyStreamErrorPrefix) + "unknown exception");
  }
}

} // namespace facebook::velox::filesystems
