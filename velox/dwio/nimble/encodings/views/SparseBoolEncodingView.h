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

#include <algorithm>

#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"

namespace facebook::nimble {

class SparseBoolEncodingView final : public TypedEncodingView<bool> {
 public:
  SparseBoolEncodingView(
      std::string_view data,
      velox::memory::MemoryPool* pool,
      const Encoding::Options& options)
      : TypedEncodingView<bool>{data, pool, options} {
    NIMBLE_CHECK_EQ(this->encodingType_, EncodingType::SparseBool);
    const char* pos = data.data() + this->dataOffset_;
    sparseValue_ = static_cast<bool>(encoding::readChar(pos));
    indices_ = detail::createTypedEncodingView<uint32_t>(
        {pos, static_cast<size_t>(data.data() + data.size() - pos)},
        pool,
        options);
    NIMBLE_CHECK_NOT_NULL(indices_);
    NIMBLE_CHECK_GT(indices_->rowCount(), 0);
    NIMBLE_CHECK_EQ(
        indices_->readAt(indices_->rowCount() - 1), this->rowCount_);
  }

 private:
  inline uint32_t lowerBoundSparseIndex(uint32_t index) const {
    uint32_t begin{0};
    uint32_t end{indices_->rowCount()};
    while (begin < end) {
      const auto mid = begin + (end - begin) / 2;
      if (indices_->readAt(mid) < index) {
        begin = mid + 1;
      } else {
        end = mid;
      }
    }
    return begin;
  }

  inline bool containsSparseIndex(uint32_t index) const {
    const auto position = lowerBoundSparseIndex(index);
    return position < indices_->rowCount() &&
        indices_->readAt(position) == index;
  }

  bool readTypedAt(uint32_t index) const final {
    NIMBLE_CHECK_LT(index, this->rowCount_);
    return containsSparseIndex(index) ? sparseValue_ : !sparseValue_;
  }

  void readPhysical(uint32_t offset, uint32_t length, bool* output)
      const final {
    this->checkReadRange(offset, length);
    std::fill(output, output + length, !sparseValue_);
    auto sparseIndex = lowerBoundSparseIndex(offset);
    const auto end = offset + length;
    while (sparseIndex < indices_->rowCount()) {
      const auto row = indices_->readAt(sparseIndex);
      if (row >= end) {
        break;
      }
      output[row - offset] = sparseValue_;
      ++sparseIndex;
    }
  }

  bool sparseValue_{false};
  std::unique_ptr<TypedEncodingView<uint32_t>> indices_;
};

} // namespace facebook::nimble
