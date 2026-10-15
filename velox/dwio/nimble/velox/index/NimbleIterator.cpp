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

// Maintains row navigation independently of stripe loading.

#include "velox/dwio/nimble/velox/index/NimbleIterator.h"

#include "velox/dwio/nimble/common/Exceptions.h"

namespace facebook::nimble {

bool NimbleIterator::hasCurrentRow() const {
  return rowRange_.has_value() && row_ < rowRange_->endRow;
}

uint32_t NimbleIterator::currentRow() const {
  NIMBLE_CHECK(hasCurrentRow(), "Iterator is not positioned on a row");
  return row_;
}

void NimbleIterator::setRowRange(std::optional<RowRange> rowRange) {
  // An unset range invalidates the iterator on reset, before a seek completes,
  // or when lookup finds no matching rows. Normal exhaustion keeps the range
  // set and row_ at its exclusive end.
  rowRange_ = rowRange;
  row_ = rowRange_.has_value() ? rowRange_->startRow : 0;
}

void NimbleIterator::advanceRows(uint32_t numRows) {
  NIMBLE_CHECK(hasCurrentRow(), "Iterator is not positioned on a row");
  NIMBLE_CHECK_LE(
      numRows, rowRange_->endRow - row_, "Cannot advance past the scan result");
  row_ += numRows;
}

} // namespace facebook::nimble
