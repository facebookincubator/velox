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

#include "velox/dwio/nimble/index/HashIndexWriter.h"

namespace facebook::nimble::index::test {

/// Test helper class for HashIndexWriter that provides access to private
/// members for testing purposes.
class HashIndexWriterTestHelper {
 public:
  explicit HashIndexWriterTestHelper(HashIndexWriter* writer)
      : writer_(writer) {}

  void setNumRows(uint32_t rows) {
    writer_->numRows_ = rows;
  }

  uint32_t numRows() const {
    return writer_->numRows_;
  }

  size_t numEntries(size_t index = 0) const {
    return writer_->accumulators_[index].entries.size();
  }

 private:
  HashIndexWriter* const writer_;
};

} // namespace facebook::nimble::index::test
