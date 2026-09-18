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

#include "velox/common/base/Nulls.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/FlatVector.h"

namespace facebook::nimble {

// Adapter for flat vectors - provides unified interface for null checking,
// index mapping, and value access. Caches raw pointers for performance.
//
// Usage:
//   FlatAdapter<int32_t> adapter(vector);
//   if (adapter.hasNulls() && adapter.isNullAt(i)) { ... }
//   auto value = adapter.valueAt(i);
//
// Template parameter T defaults to int8_t for cases where only null checking
// and index mapping are needed (no value access).
template <typename T = int8_t>
class FlatAdapter {
  static constexpr auto kIsBool = std::is_same_v<T, bool>;

 public:
  explicit FlatAdapter(const velox::VectorPtr& vector)
      : vector_{vector}, nulls_{vector->rawNulls()} {
    if constexpr (!kIsBool) {
      if (auto casted = vector->asFlatVector<T>()) {
        values_ = casted->rawValues();
      }
    }
  }

  bool hasNulls() const {
    return vector_->mayHaveNulls();
  }

  bool isNullAt(velox::vector_size_t index) const {
    return velox::bits::isBitNull(nulls_, index);
  }

  T valueAt(velox::vector_size_t index) const {
    if constexpr (kIsBool) {
      return static_cast<const velox::FlatVector<T>*>(vector_.get())
          ->valueAtFast(index);
    } else {
      return values_[index];
    }
  }

  velox::vector_size_t index(velox::vector_size_t index) const {
    return index;
  }

 private:
  const velox::VectorPtr& vector_;
  const uint64_t* nulls_;
  const T* values_{nullptr};
};

// Adapter for decoded vectors - provides unified interface for null checking,
// index mapping, and value access.
//
// Usage:
//   DecodedAdapter<int32_t> adapter(decodedVector);
//   if (adapter.hasNulls() && adapter.isNullAt(i)) { ... }
//   auto baseIndex = adapter.index(i);
//   auto value = adapter.valueAt(i);
//
// Template parameter T defaults to int8_t for cases where only null checking
// and index mapping are needed (no value access).
// Template parameter IgnoreNulls can be set to true to skip null checks.
template <typename T = int8_t, bool IgnoreNulls = false>
class DecodedAdapter {
 public:
  explicit DecodedAdapter(const velox::DecodedVector& decoded)
      : decoded_(decoded) {}

  bool hasNulls() const {
    if constexpr (IgnoreNulls) {
      return false;
    } else {
      return decoded_.mayHaveNulls();
    }
  }

  bool isNullAt(velox::vector_size_t index) const {
    if constexpr (IgnoreNulls) {
      return false;
    } else {
      return decoded_.isNullAt(index);
    }
  }

  T valueAt(velox::vector_size_t index) const {
    return decoded_.valueAt<T>(index);
  }

  velox::vector_size_t index(velox::vector_size_t index) const {
    return decoded_.index(index);
  }

 private:
  const velox::DecodedVector& decoded_;
};

} // namespace facebook::nimble
