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

#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "velox/common/memory/Memory.h"
#include "velox/core/PlanNode.h"
#include "velox/type/Type.h"
#include "velox/vector/ComplexVector.h"

namespace facebook::velox::serializer {

/// KeyDecoder decodes byte-comparable keys produced by KeyEncoder back into
/// Velox vectors. The decoded RowVector contains only the key columns, in the
/// same order passed to create().
///
/// Floating point values round-trip through KeyEncoder's canonicalized
/// representation. In particular, all NaNs decode as canonical quiet NaNs and
/// -0.0 decodes as +0.0.
class KeyDecoder {
 public:
  /// Factory method to create a KeyDecoder instance.
  ///
  /// @param keyColumns Names of columns included in the encoded key, in order
  /// @param inputType Row type used to look up column types by name
  /// @param sortOrders Sort order for each key column (ascending/descending,
  ///                   nulls first/last)
  /// @param pool Memory pool for allocations
  /// @return Unique pointer to a new KeyDecoder instance
  static std::unique_ptr<KeyDecoder> create(
      const std::vector<std::string>& keyColumns,
      RowTypePtr inputType,
      std::vector<core::SortOrder> sortOrders,
      memory::MemoryPool* pool);

  /// Decodes byte-comparable keys back into a RowVector of key columns.
  RowVectorPtr decode(std::span<const std::string_view> encodedKeys) const;

  /// Convenience overload for string-backed encoded keys.
  RowVectorPtr decode(std::span<const std::string> encodedKeys) const;

  /// Returns the decoded output type. This contains only the key columns in
  /// the order passed to create().
  const RowTypePtr& outputType() const {
    return outputType_;
  }

  /// Returns the sort orders for each key column.
  const std::vector<core::SortOrder>& sortOrders() const {
    return sortOrders_;
  }

 private:
  KeyDecoder(
      const std::vector<std::string>& keyColumns,
      RowTypePtr inputType,
      std::vector<core::SortOrder> sortOrders,
      memory::MemoryPool* pool);

  const RowTypePtr outputType_;
  const std::vector<core::SortOrder> sortOrders_;
  memory::MemoryPool* const pool_;
};

} // namespace facebook::velox::serializer
