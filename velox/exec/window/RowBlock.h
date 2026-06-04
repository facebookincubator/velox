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
#pragma once

#include "velox/common/base/Exceptions.h"
#include "velox/vector/ComplexVector.h"

#include <utility>

namespace facebook::velox::exec::window {

/// Represents a contiguous row range [startRow, endRow) from an input vector.
struct RowBlock {
  /// Creates a range over 'input' and checks that [startRow, endRow) is a valid
  /// row range within 'input'.
  RowBlock(RowVectorPtr input, vector_size_t startRow, vector_size_t endRow)
      : input(std::move(input)), startRow(startRow), endRow(endRow) {
    VELOX_CHECK_NOT_NULL(this->input, "Input vector must not be null");
    VELOX_CHECK_LE(
        startRow, endRow, "startRow must be less than or equal to endRow");
    VELOX_CHECK_LE(
        endRow,
        this->input->size(),
        "endRow must be less than or equal to input size");
  }

  /// Input vector that owns the rows in this range.
  RowVectorPtr input;

  /// First row in 'input', inclusive.
  vector_size_t startRow;

  /// Last row in 'input', exclusive.
  vector_size_t endRow;

  /// Number of rows in the range.
  vector_size_t size() const {
    return endRow - startRow;
  }
};

} // namespace facebook::velox::exec::window
