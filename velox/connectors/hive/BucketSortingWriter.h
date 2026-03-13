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
#include <vector>

#include "velox/common/base/SpillConfig.h"
#include "velox/exec/PrefixSort.h"
#include "velox/type/Type.h"
#include "velox/vector/ComplexVector.h"

namespace facebook::velox::dwio::common {
class Writer;
} // namespace facebook::velox::dwio::common

namespace facebook::velox::connector::hive {

struct HiveWriterInfo;

/// Encapsulates bucket sort configuration and wraps format writers with
/// SortingWriter when bucket sorting is enabled. Sits at the same layer as
/// PartitionWriter — owns the sort configuration and provides the wrapping
/// logic used during writer creation.
class BucketSortingWriter {
 public:
  /// @param dataType Schema of the data columns (non-partition columns).
  /// @param sortColumnIndices Indices of sort columns within dataType.
  /// @param sortCompareFlags Sort order flags for each sort column.
  /// @param finishTimeSliceLimitMs Time slice limit for finish processing.
  /// @param maxOutputRows Maximum rows per output batch from SortingWriter.
  /// @param maxOutputBytes Maximum bytes per output batch from SortingWriter.
  /// @param prefixSortConfig Prefix sort configuration.
  /// @param spillConfig Spill configuration (nullptr if spilling disabled).
  BucketSortingWriter(
      RowTypePtr dataType,
      std::vector<column_index_t> sortColumnIndices,
      std::vector<CompareFlags> sortCompareFlags,
      uint64_t finishTimeSliceLimitMs,
      uint64_t maxOutputRows,
      uint64_t maxOutputBytes,
      common::PrefixSortConfig prefixSortConfig,
      const common::SpillConfig* spillConfig);

  /// Returns true if bucket sorting is enabled.
  bool enabled() const {
    return !sortColumnIndices_.empty();
  }

  /// Returns the finish time slice limit in milliseconds.
  uint64_t finishTimeSliceLimitMs() const {
    return finishTimeSliceLimitMs_;
  }

  /// Wraps a format writer with SortingWriter if sorting is enabled.
  /// Returns the writer unchanged if sorting is not enabled.
  std::unique_ptr<dwio::common::Writer> wrap(
      HiveWriterInfo* writerInfo,
      std::unique_ptr<dwio::common::Writer> writer);

 private:
  const RowTypePtr dataType_;
  const std::vector<column_index_t> sortColumnIndices_;
  const std::vector<CompareFlags> sortCompareFlags_;
  const uint64_t finishTimeSliceLimitMs_;
  const uint64_t maxOutputRows_;
  const uint64_t maxOutputBytes_;
  const common::PrefixSortConfig prefixSortConfig_;
  const common::SpillConfig* const spillConfig_;
};

} // namespace facebook::velox::connector::hive
