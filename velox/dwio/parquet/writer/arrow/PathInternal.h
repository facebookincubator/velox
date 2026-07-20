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

// Adapted from Apache Arrow.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "arrow/result.h"
#include "arrow/status.h"

#include "velox/dwio/parquet/writer/arrow/Platform.h"

namespace arrow {

class Array;

} // namespace arrow

namespace facebook::velox::parquet::arrow {

struct ArrowWriteContext;

namespace arrow {

// This files contain internal implementation details and should not be.
// Considered part of the public API.

// The MultipathLevelBuilder is intended to fully support all Arrow nested
// types. That map to parquet types (i.e. Everything but Unions).
//

/// \brief Half open range of elements in an array.
struct ElementRange {
  /// Upper bound of range (inclusive)
  int64_t start;
  /// Upper bound of range (exclusive)
  int64_t end;

  bool empty() const {
    return start == end;
  }

  int64_t size() const {
    return end - start;
  }
};

/// \brief Result for a single leaf array when running the builder on the.
/// Its root.
struct MultipathLevelBuilderResult {
  /// \brief The Array containing only the values to write (after all nesting.
  /// Has been processed.
  ///
  /// No additional processing is done on this array (it is copied as is when.
  /// Visited via a DFS).
  std::shared_ptr<::arrow::Array> leafArray;

  /// \brief Might be null.
  const int16_t* defLevels = nullptr;

  /// \brief  Might be null.
  const int16_t* repLevels = nullptr;

  /// \brief Number of items (int16_t) contained in def/rep_levels when present.
  int64_t defRepLevelCount = 0;

  /// \brief Contains element ranges of the required visiting on the.
  /// Descendants of the final list ancestor for any leaf node.
  ///
  /// The algorithm will attempt to consolidate visited ranges into.
  /// The smallest number possible.
  ///
  /// This data is necessary to pass along because after producing.
  /// Def-rep levels for each leaf array it is impossible to determine.
  /// Which values have to be sent to parquet when a null list value.
  /// In a nullable ListArray is non-empty.
  ///
  /// This allows for the parquet writing to determine which values ultimately.
  /// Needs to be written.
  std::vector<ElementRange> postListVisitedElements;

  /// Whether the leaf array is nullable.
  bool leafIsNullable;
};

/// \brief Logic for being able to write out nesting (rep/def level) data that.
/// Is needed for writing to parquet.
class PARQUET_EXPORT MultipathLevelBuilder {
 public:
  /// \brief A callback function that will receive results from the call to.
  /// Write(...) below.  The MultipathLevelBuilderResult passed in will.
  /// Only remain valid for the function call (i.e. storing it and relying.
  /// For its data to be consistent afterwards will result in undefined.
  /// Behavior.
  using CallbackFunction =
      std::function<::arrow::Status(const MultipathLevelBuilderResult&)>;

  /// \brief Determine rep/def level information for the array.
  ///
  /// The callback will be invoked for each leaf Array that is a.
  /// Descendant of array.  Each leaf array is processed in a depth.
  /// First traversal-order.
  ///
  /// \param[in] array The array to process.
  /// \param[in] array_field_nullable Whether the algorithm should consider.
  ///   The the array column as nullable (as determined by its type's parent.
  ///   Field).
  /// \param[in, out] context for use when allocating memory, etc.
  /// \param[out] write_leaf_callback Callback to receive results.
  /// There will be one call to the write_leaf_callback for each leaf node.
  static ::arrow::Status write(
      const ::arrow::Array& array,
      bool arrayFieldNullable,
      ArrowWriteContext* context,
      CallbackFunction writeLeafCallback);

  /// \brief Construct a new instance of the builder.
  ///
  /// \param[in] array The array to process.
  /// \param[in] array_field_nullable Whether the algorithm should consider.
  ///   The the array column as nullable (as determined by its type's parent.
  ///   Field).
  static ::arrow::Result<std::unique_ptr<MultipathLevelBuilder>> make(
      const ::arrow::Array& array,
      bool arrayFieldNullable);

  virtual ~MultipathLevelBuilder() = default;

  /// \brief Returns the number of leaf columns that need to be written.
  /// To Parquet.
  virtual int getLeafCount() const = 0;

  /// \brief Calls write_leaf_callback with the MultipathLevelBuilderResult.
  /// Corresponding to |leaf_index|.
  ///
  /// \param[in] leaf_index The index of the leaf column to write.  Must be in.
  /// The range [0, GetLeafCount()]. \param[in, out] context for use when.
  /// Allocating memory, etc. \param[out] write_leaf_callback Callback to.
  /// Receive the result.
  virtual ::arrow::Status write(
      int leafIndex,
      ArrowWriteContext* context,
      CallbackFunction writeLeafCallback) = 0;
};

} // namespace arrow
} // namespace facebook::velox::parquet::arrow
