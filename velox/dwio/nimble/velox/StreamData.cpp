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

#include "velox/dwio/nimble/velox/StreamData.h"

namespace facebook::nimble {

void NullsStreamData::ensureAdditionalNullsCapacity(
    bool mayHaveNulls,
    uint64_t additionalSize) {
  if (additionalSize == 0) {
    return;
  }

  if (mayHaveNulls || hasNulls_) {
    const auto newSize = bufferedCount_ + additionalSize;
    ensureDataCapacity(nonNulls_, newSize);
    if (!hasNulls_) {
      hasNulls_ = true;
      std::fill(nonNulls_.data(), nonNulls_.data() + bufferedCount_, true);
      nonNulls_.update_size(bufferedCount_);
    }
    if (!mayHaveNulls) {
      std::fill(
          nonNulls_.data() + bufferedCount_, nonNulls_.data() + newSize, true);
      nonNulls_.update_size(newSize);
    }
  }
  bufferedCount_ += additionalSize;
}

} // namespace facebook::nimble
