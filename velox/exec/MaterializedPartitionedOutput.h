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

#include "velox/core/Expressions.h"
#include "velox/core/PlanNode.h"
#include "velox/exec/MaterializedOutputBufferManager.h"
#include "velox/exec/Operator.h"
#include "velox/row/CompactRow.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/SelectivityVector.h"

namespace facebook::velox::exec {

/// Serializes PartitionedOutput rows as CompactRow RowGroups and writes them
/// through a MaterializedOutputBuffer.
class MaterializedPartitionedOutput : public Operator {
 public:
  MaterializedPartitionedOutput(
      int32_t operatorId,
      DriverCtx* ctx,
      const std::shared_ptr<const core::PartitionedOutputNode>& planNode,
      const std::shared_ptr<MaterializedOutputBufferManager>& manager);

  void addInput(RowVectorPtr input) override;

  RowVectorPtr getOutput() override;

  bool needsInput() const override {
    return !finished_ && blockingReason_ == BlockingReason::kNotBlocked;
  }

  void noMoreInput() override;

  BlockingReason isBlocked(ContinueFuture* future) override;

  bool isFinished() override;

  void close() override;

 private:
  // Projects output columns and loads lazy vectors before serialization.
  void initializeInput(RowVectorPtr input);

  // Computes one destination for each input row before replication.
  void computePartitions(const RowVector& input, int32_t numRows);

  // Appends the current output rows to the serialized staging buffers.
  void serializeRows(row::CompactRow& compactRow, int32_t numRows);

  // Serializes fixed-width rows directly into pre-sized staging storage.
  void serializeFixedWidthRows(row::CompactRow& compactRow, int32_t numRows);

  // Sizes variable-width rows before appending them to staging storage.
  void serializeVariableWidthRows(row::CompactRow& compactRow, int32_t numRows);

  // Grows staging storage while preserving rows already serialized into it.
  void ensureFlatBufferCapacity(int64_t additionalBytes);

  // Finds rows with null partition keys for replicate-nulls-and-any.
  void collectNullRows(const RowVector& input, int32_t numRows);

  // Selects null-key rows and the first row seen by this operator exactly
  // once.
  std::vector<int32_t> selectRowsToReplicate(int32_t numRows);

  // Replicates row metadata across destinations without copying row bytes.
  void appendReplicaEntries(
      int32_t serializeStartRow,
      const std::vector<int32_t>& rowsToReplicate);

  // Adds replica metadata for the rows selected from the current input batch.
  void expandReplicateRows(int32_t serializeStartRow, int32_t numRows);

  // Groups staged rows by destination, emits bounded RowGroups, and resets.
  void flushBatch();

  // Encodes selected rows as one CompactRow RowGroup and enqueues it.
  void flushRowGroup(int32_t partition, std::vector<int32_t>& rowIndices);

  // Flushes local rows and lets the last peer commit the shared buffer.
  void finish();

  // Publishes buffer and sink counters as operator runtime statistics.
  void recordBufferStats();

  bool shouldReplicate() const {
    return !broadcast_ && replicateNullsAndAny_ && numDestinations_ > 1;
  }

  const int32_t numDestinations_;
  const bool broadcast_;
  const std::vector<column_index_t> outputChannels_;
  const std::vector<column_index_t> keyChannels_;
  std::unique_ptr<core::PartitionFunction> partitionFunction_;
  const bool replicateNullsAndAny_;
  const std::shared_ptr<MaterializedOutputBuffer> buffer_;
  const int64_t targetSizeInBytes_;
  const int64_t rowGroupMaxBytes_;
  const std::optional<int32_t> fixedRowSize_;

  BlockingReason blockingReason_{BlockingReason::kNotBlocked};
  ContinueFuture future_;
  bool finished_{false};
  bool isLastDriver_{false};

  RowVectorPtr output_;
  std::vector<uint32_t> partitions_;

  SelectivityVector rows_;
  SelectivityVector nullRows_;
  std::vector<DecodedVector> decodedVectors_;
  bool replicatedAny_{false};

  int32_t rowCount_{0};
  int64_t flatBufferSize_{0};
  BufferPtr flatBuffer_;
  std::vector<int64_t> rowOffsets_;
  std::vector<int32_t> rowSizes_;
  std::vector<uint32_t> rowPartitions_;
};

} // namespace facebook::velox::exec
