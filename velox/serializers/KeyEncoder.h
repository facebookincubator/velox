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

#include <functional>

#include "velox/common/base/Exceptions.h"
#include "velox/common/memory/Scratch.h"
#include "velox/core/PlanNode.h"
#include "velox/type/Type.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/DecodedVector.h"

namespace facebook::velox::serializer {

/// A single index bound (lower or upper) with inclusive flag.
/// Used to represent either a lower or upper bound for index filtering.
/// The bound vector must be non-null with at least 1 row. To represent an
/// unbounded range, use std::nullopt in the host IndexBounds instead of a
/// null bound vector.
struct IndexBound {
  /// Bound values. For single-row bounds, this contains exactly 1 row.
  /// For multi-row bounds (batch processing), this may contain N rows.
  /// Must be non-null with at least 1 row.
  RowVectorPtr bound;
  /// Whether this bound is inclusive. If true, the bound value is included
  /// in the range; if false, it's excluded.
  bool inclusive{true};
};

/// Represents bounds for index-based filtering with optional lower and upper
/// bounds. Used to define a range of values for index columns that can be
/// encoded into byte-comparable keys for efficient filtering.
struct IndexBounds {
  /// The top-level column names that form the index. These columns must exist
  /// in the input data and define the order of columns in the encoded key.
  std::vector<std::string> indexColumns;
  /// Lower bound for the index range. std::nullopt means unbounded (no lower
  /// bound).
  std::optional<IndexBound> lowerBound;
  /// Upper bound for the index range. std::nullopt means unbounded (no upper
  /// bound).
  std::optional<IndexBound> upperBound;

  /// Sets both bounds and validates them. Throws if validation fails.
  /// The indexColumns must be set before calling this method.
  void set(IndexBound lower, IndexBound upper);

  /// Clears both lower and upper bounds. The indexColumns are preserved.
  void clear();

  /// Returns the type from the bound vectors.
  /// Extracts the type from lowerBound if available, otherwise from upperBound.
  /// At least one bound must be present (non-null).
  TypePtr type() const;

  /// Returns the number of rows in the bounds. Requires at least one bound
  /// to be present. If both bounds exist, they must have the same row count.
  vector_size_t numRows() const;

  /// Validates that the bounds are well-formed:
  /// - At least one bound (lower or upper) must be present (non-null)
  /// - Each non-null bound must have at least 1 row
  /// - If both bounds are present, they must have the same number of rows
  /// Returns true if valid, false otherwise.
  bool validate() const;

  /// Returns a human-readable string representation of the bounds.
  std::string toString() const;
};

/// Encoded representation of index bounds as byte-comparable strings.
/// These keys can be compared lexicographically to perform range filtering.
struct EncodedKeyBounds {
  /// Encoded lower bound key. If present, represents the minimum key value
  /// (inclusive or exclusive based on IndexBound.inclusive).
  std::optional<std::string> lowerKey;
  /// Encoded upper bound key. If present, represents the maximum key value
  /// (inclusive or exclusive based on IndexBound.inclusive).
  std::optional<std::string> upperKey;
};

/// KeyEncoder encodes multi-column keys into byte-comparable strings that
/// preserve the sort order defined by the column types and sort orders.
/// This enables efficient range-based filtering and comparison of composite
/// keys.
///
/// The encoding is designed such that:
/// - Lexicographic comparison of encoded keys matches the logical comparison
///   of the original values
/// - Null values are sorted according to the specified sort order (nulls
///   first/last)
/// - Ascending/descending sort orders are respected
/// - Only scalar types are supported (e.g., integers, floats, strings,
///   booleans, dates). Complex types (arrays, maps, rows) are not supported.
///
/// Example usage:
///   auto keyEncoder = KeyEncoder::create(
///       {"col1", "col2"}, rowType, sortOrders, pool);
///   std::vector<std::string_view> encodedKeys;
///   HashStringAllocator allocator(pool);
///   keyEncoder->encode(inputVector, encodedKeys,
///       [&allocator](size_t size) { return allocator.allocate(size); });
class KeyEncoder {
 public:
  /// Factory method to create a KeyEncoder instance.
  ///
  /// @param keyColumns Names of columns to include in the encoded key, in order
  /// @param inputType Row type of the input data containing these columns
  /// @param sortOrders Sort order for each key column (ascending/descending,
  ///                   nulls first/last)
  /// @param pool Memory pool for allocations
  /// @return Unique pointer to a new KeyEncoder instance
  static std::unique_ptr<KeyEncoder> create(
      std::vector<std::string> keyColumns,
      RowTypePtr inputType,
      std::vector<core::SortOrder> sortOrders,
      memory::MemoryPool* pool);

  /// Type alias for buffer allocator function.
  /// Takes estimated size in bytes and returns a pointer to the allocated
  /// buffer.
  using BufferAllocator = std::function<void*(size_t)>;

  /// Encodes the key columns from the input vector into byte-comparable keys.
  ///
  /// Each row in the input produces one encoded key string. The keys can be
  /// compared lexicographically, and the comparison result will match the
  /// logical comparison based on the specified sort orders.
  ///
  /// @tparam Container A container type for std::string_view that supports
  ///                   reserve(), size(), and emplace_back() operations.
  /// @param input Input vector containing rows to encode
  /// @param encodedKeys Output container to store the encoded key strings
  ///                    (views into allocated buffer)
  /// @param bufferAllocator Allocator function that takes estimated size and
  ///                        returns pointer to allocated buffer
  template <typename Container>
  void encode(
      const VectorPtr& input,
      Container& encodedKeys,
      const BufferAllocator& bufferAllocator);

  /// Encodes index bounds into byte-comparable boundary keys.
  ///
  /// The implementation normalizes all bounds to half-open interval format
  /// [lower_bound, upper_bound) to ease range scan processing:
  /// - Lower bounds: Always converted to inclusive
  ///   - Exclusive lower bound (x > 5) → incremented to inclusive (x >= 6)
  ///   - Inclusive lower bound (x >= 5) → stays as is
  /// - Upper bounds: Always converted to exclusive
  ///   - Inclusive upper bound (x <= 10) → incremented to exclusive (x < 11)
  ///   - Exclusive upper bound (x < 10) → stays as is
  ///
  /// Increment failure handling:
  /// - Lower bound increment fails → returns std::nullopt (encoding failed,
  ///   cannot establish valid range start)
  /// - Upper bound increment fails → upperKey set to std::nullopt (treated as
  ///   unbounded upper range)
  ///
  /// Increment fails when values are at their maximum (e.g., INT_MAX, strings
  /// with all \xFF characters, or nulls in NULLS_LAST ordering).
  ///
  /// For multi-row bounds, returns a vector with one EncodedKeyBounds per row.
  /// Each row is processed independently.
  /// Encodes index bounds into byte-comparable key strings.
  /// Takes an IndexBounds containing lower and/or upper bounds and encodes them
  /// into EncodedKeyBounds for efficient range comparison.
  /// Throws if any lower bound fails to bump up (for exclusive bounds).
  /// For upper bound bump up failures, the upperKey is set to std::nullopt
  /// (unbounded).
  std::vector<EncodedKeyBounds> encodeIndexBounds(
      const IndexBounds& indexBounds);

  /// Returns the sort orders for each index column.
  const std::vector<core::SortOrder>& sortOrders() const {
    return sortOrders_;
  }

 private:
  KeyEncoder(
      std::vector<std::string> keyColumns,
      RowTypePtr inputType,
      std::vector<core::SortOrder> sortOrders,
      memory::MemoryPool* pool);

  uint64_t estimateEncodedSize();

  // Encodes a RowVector and returns encoded keys as strings.
  // Each row in the input vector produces one encoded key string.
  std::vector<std::string> encode(const RowVectorPtr& input);

  // Creates a new row vector with the key columns incremented by 1 for multiple
  // rows. Returns nullptr if any row fails to increment (all key columns
  // overflow), otherwise returns RowVectorPtr with incremented values.
  RowVectorPtr createIncrementedBounds(const RowVectorPtr& bounds) const;

  // Encodes a single column for all rows.
  void encodeColumn(
      const DecodedVector& decodedVector,
      vector_size_t numRows,
      bool descending,
      bool nullLast,
      std::vector<char*>& rowOffsets);

  const RowTypePtr inputType_;
  const std::vector<core::SortOrder> sortOrders_;
  const std::vector<vector_size_t> keyChannels_;
  memory::MemoryPool* const pool_;

  // Reusable buffers.
  DecodedVector decodedVector_;
  std::vector<DecodedVector> childDecodedVectors_;
  std::vector<vector_size_t> encodedSizes_;
  Scratch scratch_;
};

// Template implementation
template <typename Container>
void KeyEncoder::encode(
    const VectorPtr& input,
    Container& encodedKeys,
    const BufferAllocator& bufferAllocator) {
  VELOX_CHECK_GT(input->size(), 0);
  SCOPE_EXIT {
    encodedSizes_.clear();
  };
  decodedVector_.decode(*input, /*loadLazy=*/true);
  const auto* rowBase = decodedVector_.base()->asChecked<RowVector>();
  const auto& children = rowBase->children();
  for (auto i = 0; i < keyChannels_.size(); ++i) {
    childDecodedVectors_[i].decode(*children[keyChannels_[i]]);
  }
  const auto totalBytes = estimateEncodedSize();
  auto* const allocated = static_cast<char*>(bufferAllocator(totalBytes));

  const auto numRows = input->size();
  const auto numKeys = keyChannels_.size();

  // Compute buffer start offsets for each row
  std::vector<char*> rowOffsets(numRows);
  rowOffsets[0] = allocated;
  for (auto row = 1; row < numRows; ++row) {
    rowOffsets[row] = rowOffsets[row - 1] + encodedSizes_[row - 1];
  }

  // Encode column-by-column for better cache locality
  for (auto i = 0; i < numKeys; ++i) {
    const bool nullLast = !sortOrders_[i].isNullsFirst();
    const bool descending = !sortOrders_[i].isAscending();
    const auto& decodedVector = childDecodedVectors_[i];

    // Encode column data for all rows (null indicator is encoded within each
    // type's encoding function)
    encodeColumn(decodedVector, numRows, descending, nullLast, rowOffsets);
  }

  // Build encoded keys string views
  encodedKeys.reserve(encodedKeys.size() + numRows);
  size_t offset{0};
  for (auto row = 0; row < numRows; ++row) {
    encodedKeys.emplace_back(allocated + offset, encodedSizes_[row]);
    offset += encodedSizes_[row];
    VELOX_CHECK_EQ(rowOffsets[row], allocated + offset);
  }
}

} // namespace facebook::velox::serializer
