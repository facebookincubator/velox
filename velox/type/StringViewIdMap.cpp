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

#include "velox/type/StringViewIdMap.h"

namespace facebook::velox {

StringViewIdMap::StringViewIdMap(int32_t capacity)
    : capacity_(std::max<int32_t>(
          64,
          bits::nextPowerOfTwo(capacity + (capacity / 2)))) {
  sizeMask_ = capacity_ - 1;
  table_.resize(capacity_ * kEntrySize);
  memset(table_.data(), 0, table_.size());
  // Rehash when three quarters full.
  maxEntries_ = capacity_ / 4 * 3;
  lastEntryOffset_ = (capacity_ - 1) * kEntrySize;
}

void StringViewIdMap::resize(int32_t newSize) {
  VELOX_CHECK_LT(
      static_cast<int64_t>(kEntrySize) * newSize,
      1 << 30,
      "Size of StringViewIdMap not to exceed 1GB");
  raw_vector<uint8_t> oldTable = std::move(table_);
  auto limit = lastEntryOffset_;
  capacity_ = newSize;
  sizeMask_ = newSize - 1;
  lastEntryOffset_ = (capacity_ - 1) * kEntrySize;
  // Rehash when 5/8 full. This is a little sooner than the initial
  // 3/4 in order to grow faster.
  maxEntries_ = capacity_ / 8 * 5;
  table_.resize(newSize * kEntrySize);
  memset(table_.data(), 0, table_.size());
  auto oldData = oldTable.data();
  auto data = table_.data();
  for (int32_t offset = 0; offset <= limit; offset += kEntrySize) {
    if (itemAt<int64_t>(oldData, offset) == kEmpty) {
      continue;
    }
    auto* view = &itemAt<StringView>(oldData, offset);
    auto newOffset = (hash1(*view) & sizeMask_) * kEntrySize;
    for (;;) {
      if (itemAt<int64_t>(data, newOffset) == 0) {
        memcpy(data + newOffset, oldData + offset, kEntrySize);
        break;
      }
      newOffset = nextOffset(newOffset);
    }
  }
}

void StringViewIdMap::clear() {
  numEntries_ = 0;
  memset(table_.data(), 0, table_.size());
}
} // namespace facebook::velox
