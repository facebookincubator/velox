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

#include "velox/dwio/nimble/writer/BufferPolicy.h"

#include <algorithm>
#include <cmath>

#include "velox/dwio/common/Range.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/writer/RawSizeUtils.h"

namespace facebook::nimble {

StripeRawSizeBufferPolicy::StripeRawSizeBufferPolicy(uint64_t stripeRawSize)
    : stripeRawSize_{stripeRawSize} {
  NIMBLE_USER_CHECK_GT(stripeRawSize_, 0, "Stripe raw size must be positive");
}

void StripeRawSizeBufferPolicy::bufferInput(velox::VectorPtr input) {
  NIMBLE_USER_CHECK_NOT_NULL(input, "Buffered input must not be null");
  const auto numRows = static_cast<uint32_t>(input->size());
  NIMBLE_USER_CHECK_GT(numRows, 0, "Buffered input must not be empty");

  velox::common::Ranges ranges;
  ranges.add(0, input->size());
  // At least one byte per row, so rows without raw bytes still fill stripes.
  const double rowRawSize = std::max(
      1.0, static_cast<double>(getRawSizeFromVector(input, ranges)) / numRows);
  uint32_t row{0};
  while (row < numRows) {
    const double remainingRawSize =
        std::max(0.0, static_cast<double>(stripeRawSize_) - bufferedRawSize_);
    const auto remainingRows = static_cast<uint64_t>(
        std::max(1.0, std::ceil(remainingRawSize / rowRawSize)));
    const auto rowsToBuffer =
        static_cast<uint32_t>(std::min<uint64_t>(remainingRows, numRows - row));
    bufferedRange_.inputs.emplace_back(input);
    bufferedRange_.rowRanges.emplace_back(row, row + rowsToBuffer);
    bufferedRawSize_ += rowsToBuffer * rowRawSize;
    row += rowsToBuffer;
    if (rowsToBuffer == remainingRows) {
      flushRange();
    }
  }
}

BufferRange StripeRawSizeBufferPolicy::flushInput() {
  if (flushQueue_.empty()) {
    if (!finalized_ || bufferedRange_.empty()) {
      return {};
    }
    flushRange();
  }
  auto range = std::move(flushQueue_.front());
  flushQueue_.pop_front();
  return range;
}

void StripeRawSizeBufferPolicy::finalize() {
  NIMBLE_CHECK(!finalized_, "finalize() must be called only once");
  finalized_ = true;
}

void StripeRawSizeBufferPolicy::flushRange() {
  flushQueue_.emplace_back(std::move(bufferedRange_));
  bufferedRange_ = {};
  bufferedRawSize_ = 0;
}

} // namespace facebook::nimble
