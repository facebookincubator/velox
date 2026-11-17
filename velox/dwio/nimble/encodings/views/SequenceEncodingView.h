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
#include <span>

#include "velox/dwio/nimble/encodings/SequenceEncoding.h"
#include "velox/dwio/nimble/encodings/views/EncodingView.h"

namespace facebook::nimble {

/// Reads a SequenceEncoding by row number. Computes each value from the base
/// and the step, so it keeps no per-row state and every read costs O(1) per
/// row.
template <typename T>
class SequenceEncodingView final : public TypedEncodingView<T> {
 public:
  using physicalType = typename TypedEncodingView<T>::physicalType;

  SequenceEncodingView(
      std::string_view data,
      velox::memory::MemoryPool* pool,
      const Encoding::Options& options)
      : TypedEncodingView<T>{data, pool, options},
        base_{SequenceEncoding<T>::readBase(data, this->dataOffset_)},
        step_{SequenceEncoding<T>::readStep(data, this->dataOffset_)} {
    NIMBLE_CHECK_EQ(this->encodingType_, EncodingType::Sequence);
  }

 private:
  T readTypedAt(uint32_t index) const final {
    NIMBLE_CHECK_LT(index, this->rowCount_);
    return static_cast<T>(SequenceEncoding<T>::valueAt(base_, step_, index));
  }

  void readPhysical(uint32_t offset, uint32_t length, physicalType* output)
      const final {
    this->checkReadRange(offset, length);
    const uint64_t first = SequenceEncoding<T>::valueAt(base_, step_, offset);
    const uint64_t step = step_;
    for (uint32_t i = 0; i < length; ++i) {
      output[i] = static_cast<physicalType>(first + i * step);
    }
  }

  // Computes the values in one loop, with one bounds check for all indices.
  void readPhysicalAt(std::span<const uint32_t> indices, physicalType* output)
      const final {
    if (indices.empty()) {
      return;
    }
    this->checkReadIndices(indices);
    for (size_t i{0}; i < indices.size(); ++i) {
      output[i] = SequenceEncoding<T>::valueAt(base_, step_, indices[i]);
    }
  }

  const physicalType base_;
  const physicalType step_;
};

} // namespace facebook::nimble
