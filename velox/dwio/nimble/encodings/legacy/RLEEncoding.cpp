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
#include "velox/dwio/nimble/encodings/legacy/RLEEncoding.h"

namespace facebook::nimble::legacy {

RLEEncoding<bool>::RLEEncoding(
    velox::memory::MemoryPool& memoryPool,
    std::string_view data,
    std::function<void*(uint32_t)> stringBufferFactory)
    : internal::RLEEncodingBase<bool, RLEEncoding<bool>>(
          memoryPool,
          data,
          stringBufferFactory) {
  initialValue_ = *reinterpret_cast<const bool*>(
      internal::RLEEncodingBase<bool, RLEEncoding<bool>>::getValuesStart());
  NIMBLE_CHECK(
      (internal::RLEEncodingBase<bool, RLEEncoding<bool>>::getValuesStart() +
       1) == data.data() + data.size(),
      "Unexpected run length encoding end");
  internal::RLEEncodingBase<bool, RLEEncoding<bool>>::reset();
}

bool RLEEncoding<bool>::nextValue() {
  value_ = !value_;
  return !value_;
}

void RLEEncoding<bool>::resetValues() {
  value_ = initialValue_;
}

void RLEEncoding<bool>::materializeBoolsAsBits(
    uint32_t rowCount,
    uint64_t* buffer,
    int begin) {
  auto rowsLeft = rowCount;
  while (rowsLeft) {
    if (rowsLeft < copiesRemaining_) {
      velox::bits::fillBits(buffer, begin, begin + rowsLeft, currentValue_);
      copiesRemaining_ -= rowsLeft;
      return;
    } else {
      velox::bits::fillBits(
          buffer, begin, begin + copiesRemaining_, currentValue_);
      begin += copiesRemaining_;
      rowsLeft -= copiesRemaining_;
      copiesRemaining_ = materializedRunLengths_.nextValue();
      currentValue_ = nextValue();
    }
  }
}

} // namespace facebook::nimble::legacy
