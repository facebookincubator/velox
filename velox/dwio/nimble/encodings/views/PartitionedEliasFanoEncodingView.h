/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include <cstdint>

#include "velox/dwio/nimble/encodings/PartitionedEliasFanoEncoding.h"
#include "velox/dwio/nimble/encodings/views/EncodingView.h"

namespace facebook::nimble {

/// Provides stateless random and contiguous access to partitioned Elias-Fano
/// values.
template <typename T>
class PartitionedEliasFanoEncodingView final : public TypedEncodingView<T> {
 public:
  using physicalType = typename TypedEncodingView<T>::physicalType;

  /// Creates a view over a partitioned Elias-Fano encoded stream.
  PartitionedEliasFanoEncodingView(
      std::string_view data,
      velox::memory::MemoryPool* pool,
      const Encoding::Options& options)
      : TypedEncodingView<T>{data, pool, options},
        encoding_{*pool, data, {}, options} {
    NIMBLE_CHECK_EQ(this->encodingType_, EncodingType::PartitionedEliasFano);
  }

 private:
  T readTypedAt(uint32_t index) const final {
    return encoding_.valueAt(index);
  }

  void readPhysical(uint32_t offset, uint32_t length, physicalType* output)
      const final {
    encoding_.materializeAt(offset, length, output);
  }

  // Searches the one partition that holds the range. The encoding checks that
  // the range stays within it, which makes the range non-decreasing.
  uint32_t typedLowerBound(uint32_t begin, uint32_t end, T value, T& found)
      const final {
    physicalType physicalFound;
    const auto row = encoding_.lowerBound(
        begin, end, static_cast<physicalType>(value), physicalFound);
    if (row < end) {
      found = static_cast<T>(physicalFound);
    }
    return row;
  }

  // Owns the parsed partition directory and one Elias-Fano list per group.
  PartitionedEliasFanoEncoding<T> encoding_;
};

} // namespace facebook::nimble
