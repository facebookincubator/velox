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

#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/writer/DecodedVectorManager.h"

namespace facebook::nimble {

class RawSizeContext {
 public:
  RawSizeContext() = default;

  DecodedVectorManager& getDecodedVectorManager() {
    return decodedVectorManager_;
  }

  void appendSize(uint64_t size) {
    columnSizes_.push_back(size);
  }

  uint64_t sizeAt(uint64_t columnIndex) const {
    NIMBLE_CHECK_LT(
        columnIndex, columnSizes_.size(), "Column index is out of range.");
    return columnSizes_.at(columnIndex);
  }

  void setSizeAt(uint64_t columnIndex, uint64_t size) {
    NIMBLE_CHECK_LT(
        columnIndex, columnSizes_.size(), "Column index is out of range.");
    columnSizes_[columnIndex] = size;
  }

  uint64_t columnCount() const {
    return columnSizes_.size();
  }

 private:
  DecodedVectorManager decodedVectorManager_;
  std::vector<uint64_t> columnSizes_;
};

} // namespace facebook::nimble
