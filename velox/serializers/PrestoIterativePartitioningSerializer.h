/*
 * Copyright (c) International Business Machines Corporation
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

#include <map>
#include <memory>
#include <vector>

#include <folly/io/IOBuf.h>

#include "velox/common/memory/ByteStream.h"
#include "velox/serializers/PrestoSerializer.h"
#include "velox/type/Type.h"
#include "velox/vector/PartitionedVector.h"

namespace facebook::velox::serializer::presto {

/// Convenience alias matching PrestoSerializer.cpp convention.
using SerdeOpts = PrestoVectorSerde::PrestoOptions;

/// Serializes a stream of RowVectors into per-partition Presto pages.
///
/// Each call to append() routes rows to their assigned partition. flush()
/// produces one Presto-format IOBuf per non-empty partition and resets the
/// internal state so the serializer can be reused for the next cycle.
class PrestoIterativePartitioningSerializer {
 public:
  PrestoIterativePartitioningSerializer(
      RowTypePtr inputType,
      uint32_t numPartitions,
      const SerdeOpts& opts,
      memory::MemoryPool* pool);

  /// Routes each row in `input` to the partition indicated by `partitions`.
  /// `partitions.size()` must equal `input->size()`.
  void append(
      const RowVectorPtr& input,
      const std::vector<uint32_t>& partitions);

  /// Serializes all buffered data into one Presto page per non-empty partition
  /// and resets internal state. Returns an empty map if nothing has been
  /// appended since the last flush.
  std::map<uint32_t, std::pair<std::unique_ptr<folly::IOBuf>, vector_size_t>>
  flush();

  /// Returns the total retained bytes of all appended input vectors.
  int64_t bytesBuffered() const {
    return bytesBuffered_;
  }

  /// Returns the total number of rows appended since the last flush.
  int64_t rowsBuffered() const {
    return rowsBuffered_;
  }

  /// Returns the number of rows buffered for the given partition.
  /// Must be called before flush(), which resets per-partition counts.
  int64_t rowsPerPartition(uint32_t partition) const {
    VELOX_DCHECK_LT(partition, numPartitions_);
    return rowsPerPartition_[partition];
  }

 private:
  std::map<uint32_t, std::pair<std::unique_ptr<folly::IOBuf>, vector_size_t>>
  flushUncompressed();
  std::map<uint32_t, std::pair<std::unique_ptr<folly::IOBuf>, vector_size_t>>
  flushCompressed();

  void flushStart(IOBufOutputStream& out, uint32_t partition, char codecMask)
      const;

  void flushFinish(
      IOBufOutputStream& out,
      uint32_t partition,
      std::streampos beginOffset,
      char codecMask,
      PrestoOutputStreamListener& listener) const;

  void flushRowChildren(
      const std::vector<PartitionedVectorPtr>& partitionedVectors,
      const RowType& rowSchema,
      const std::vector<uint32_t>& nonEmptyPartitions,
      const std::vector<IOBufOutputStream*>& outputStreams) const;

  void flushColumn(
      const std::vector<PartitionedVectorPtr>& partitionedVectors,
      const TypePtr& colType,
      const std::vector<uint32_t>& nonEmptyPartitions,
      const std::vector<IOBufOutputStream*>& outputStreams) const;

  void flushSimpleColumn(
      const std::vector<PartitionedVectorPtr>& partitionedVectors,
      const TypePtr& colType,
      const std::vector<uint32_t>& nonEmptyPartitions,
      const std::vector<IOBufOutputStream*>& outputStreams) const;

  void flushSingleSimpleVector(
      const PartitionedVectorPtr& partitionedVector,
      const std::vector<IOBufOutputStream*>& outputStreams) const;

  template <TypeKind kind>
  void flushSingleFlatVector(
      const PartitionedVectorPtr& partitionedVector,
      const std::vector<IOBufOutputStream*>& outputStreams) const;

  void flushHeader(
      std::string_view name,
      const std::vector<uint32_t>& nonEmptyPartitions,
      const std::vector<IOBufOutputStream*>& outputStreams) const;

  void flushRowCounts(
      const std::vector<uint32_t>& nonEmptyPartitions,
      const std::vector<IOBufOutputStream*>& outputStreams) const;

  void flushNulls(
      const std::vector<PartitionedVectorPtr>& partitionedVectors,
      const std::vector<uint32_t>& nonEmptyPartitions,
      const std::vector<IOBufOutputStream*>& outputStreams) const;

  template <typename T>
  void flushFlatValues(
      const T* partitionedValues,
      const uint64_t* rawNulls,
      const vector_size_t* partitionOffsets,
      const std::vector<IOBufOutputStream*>& outputStreams) const;

  void flushSequentialOffsets(
      const std::vector<uint32_t>& nonEmptyPartitions,
      const std::vector<IOBufOutputStream*>& outputStreams) const;

  RowTypePtr type_;
  uint32_t numPartitions_;
  SerdeOpts opts_;
  memory::MemoryPool* pool_;

  /// Cumulative row count per partition across all appended batches.
  std::vector<vector_size_t> rowsPerPartition_;

  /// Number of top-level columns in `type_`.
  uint32_t numColumns_{0};

  std::vector<PartitionedVectorPtr> partitionedRowVectors_;

  int64_t bytesBuffered_{0};
  int64_t rowsBuffered_{0};

  /// Per-column, per-partition exact byte counts computed during flush.
  std::vector<std::vector<int64_t>> flushSizes_;
};

} // namespace facebook::velox::serializer::presto
