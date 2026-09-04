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

#include "velox/buffer/Buffer.h"
#include "velox/common/base/BitUtil.h"
#include "velox/common/base/Nulls.h"
#include "velox/vector/TypeAliases.h"

namespace facebook::velox {

/// Helper struct to lazily initialize nulls buffer on first null.
struct NullsBuilder {
  NullsBuilder(vector_size_t size, memory::MemoryPool* pool)
      : size_{size}, pool_{pool} {}

  /// Marks specified row as null. Allocates and initializes null buffer if this
  /// is the first null.
  void setNull(vector_size_t row) {
    allocate();
    bits::setNull(rawNulls_, row, true);
  }

  /// Marks every row that is null in 'nulls' as null. 'nulls' may be nullptr,
  /// meaning there is nothing to add. Allocates and initializes the null buffer
  /// on the first non-null 'nulls'.
  void addNulls(const uint64_t* nulls) {
    if (nulls == nullptr) {
      return;
    }
    allocate();
    bits::andBits(rawNulls_, nulls, 0, size_);
  }

  /// Returns nulls buffer or nullptr if no nulls were added (e.g. setNull was
  /// never called).
  BufferPtr build() const {
    return nulls_;
  }

 private:
  void allocate() {
    if (nulls_ == nullptr) {
      nulls_ = AlignedBuffer::allocate<bool>(size_, pool_, bits::kNotNull);
      rawNulls_ = nulls_->asMutable<uint64_t>();
    }
  }

  const vector_size_t size_;
  memory::MemoryPool* pool_;
  BufferPtr nulls_{nullptr};
  uint64_t* rawNulls_{nullptr};
};
} // namespace facebook::velox
