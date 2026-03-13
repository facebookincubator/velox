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
#include <memory>
#include <vector>

#include "velox/common/memory/RawVector.h"
#include "velox/connectors/hive/HiveWriterTypes.h"
#include "velox/connectors/hive/PartitionWriterInterface.h"
#include "velox/vector/ComplexVector.h"

namespace facebook::velox::connector::hive {

/// Routes rows to the correct partition writer based on partition and bucket
/// IDs. Owns a collection of partition writers and handles the fan-out of
/// input rows to writers by partition/bucket assignment.
class PartitionWriter {
 public:
  /// Factory to create a partition writer for a given writer ID and index.
  using WriterFactory = std::function<std::unique_ptr<PartitionWriterInterface>(
      const HiveWriterId& id,
      uint32_t writerIndex)>;

  /// Callback invoked after a new writer is created.
  using OnWriterCreated =
      std::function<void(uint32_t writerIndex, const HiveWriterId& id)>;

  /// @param maxOpenWriters Maximum number of open writers allowed.
  /// @param dataChannels Column indices for data columns to write.
  /// @param dataType Schema of the data columns (non-partition columns).
  /// @param writerFactory Factory to create per-partition writers.
  /// @param pool Memory pool for partition row index buffers.
  PartitionWriter(
      uint32_t maxOpenWriters,
      const std::vector<column_index_t>& dataChannels,
      RowTypePtr dataType,
      WriterFactory writerFactory,
      memory::MemoryPool* pool);

  /// Write all rows to a single writer identified by 'id'.
  void write(const HiveWriterId& id, RowVectorPtr input);

  /// Route rows to writers by partition/bucket IDs.
  void write(
      RowVectorPtr input,
      const raw_vector<uint64_t>& partitionIds,
      const std::vector<uint32_t>& bucketIds,
      bool isPartitioned,
      bool isBucketed);

  /// Flush buffered data with time-slicing. Returns true when complete.
  bool finish(uint64_t timeSliceLimitMs);

  /// Close all writers after successful completion.
  void close();

  /// Abort all writers for error cleanup.
  void abort();

  /// Ensure a writer exists for the given ID. Returns writer index.
  uint32_t ensureWriter(const HiveWriterId& id);

  /// Access the underlying writers.
  const std::vector<std::unique_ptr<PartitionWriterInterface>>& writers() const {
    return writers_;
  }

  /// Set callback invoked when a new writer is created.
  void setOnWriterCreated(OnWriterCreated callback) {
    onWriterCreated_ = std::move(callback);
  }

 private:
  /// Extract data columns from input and write to the specified writer.
  void writeToWriter(size_t index, RowVectorPtr input);

  /// Compute the writer ID for a given row from partition/bucket IDs.
  HiveWriterId getWriterId(
      size_t row,
      const raw_vector<uint64_t>& partitionIds,
      const std::vector<uint32_t>& bucketIds,
      bool isPartitioned,
      bool isBucketed) const;

  /// Split input rows by partition/bucket and ensure writers exist.
  void splitInputRowsAndEnsureWriters(
      const raw_vector<uint64_t>& partitionIds,
      const std::vector<uint32_t>& bucketIds,
      bool isPartitioned,
      bool isBucketed);

  /// Record a row index for a specific partition.
  void
  updatePartitionRows(uint32_t index, vector_size_t numRows, vector_size_t row);

  const uint32_t maxOpenWriters_;
  const std::vector<column_index_t> dataChannels_;
  const RowTypePtr dataType_;
  WriterFactory writerFactory_;
  memory::MemoryPool* pool_;

  /// Map from writer ID to index in writers_.
  folly::F14FastMap<HiveWriterId, uint32_t, HiveWriterIdHasher, HiveWriterIdEq>
      writerIndexMap_;

  /// Per-partition (or per-partition+bucket) writers.
  std::vector<std::unique_ptr<PartitionWriterInterface>> writers_;

  /// Partition row index buffers, indexed by writer index.
  std::vector<BufferPtr> partitionRows_;
  std::vector<vector_size_t*> rawPartitionRows_;
  std::vector<vector_size_t> partitionSizes_;

  OnWriterCreated onWriterCreated_;
};

} // namespace facebook::velox::connector::hive
