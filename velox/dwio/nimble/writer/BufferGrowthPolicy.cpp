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
#include "velox/dwio/nimble/writer/BufferGrowthPolicy.h"

#include <cmath>

namespace facebook::nimble {

uint64_t DefaultInputBufferGrowthPolicy::getExtendedCapacity(
    uint64_t newSize,
    uint64_t capacity) const {
  // Short circuit when we don't need to grow further.
  if (newSize <= capacity) {
    return capacity;
  }

  auto iter = rangeConfigs_.upper_bound(newSize);
  if (iter == rangeConfigs_.begin()) {
    return minCapacity_;
  }

  // Move to the range that actually contains the newSize.
  --iter;
  // The sizes are item counts, hence we should really not run into overflow or
  // precision loss.
  // TODO: We determine the growth factor only once and grow the
  // capacity until it suffices. This doesn't matter that much in practice when
  // capacities start from the min capacity and the range boundaries are
  // aligned. We could start the capacity at the range boundaries instead, after
  // some testing.
  double extendedCapacity = folly::to<double>(std::max(capacity, minCapacity_));
  while (extendedCapacity < newSize) {
    extendedCapacity *= iter->second;
  }
  return folly::to<uint64_t>(std::floor(extendedCapacity));
}

} // namespace facebook::nimble
