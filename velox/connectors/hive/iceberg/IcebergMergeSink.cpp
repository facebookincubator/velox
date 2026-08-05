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

#include "velox/connectors/hive/iceberg/IcebergMergeSink.h"

#include "velox/common/base/Exceptions.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::connector::hive::iceberg {

namespace {

// Returns the operation TINYINT child as a flat vector. Throws if the
// operation column at `channel` is not a flat TINYINT.
const FlatVector<int8_t>* asOperationFlatVector(
    const RowVectorPtr& input,
    column_index_t channel) {
  const auto& column = input->childAt(channel);
  VELOX_USER_CHECK_NOT_NULL(column, "operation column is null");
  const auto* flat = column->loadedVector()->asFlatVector<int8_t>();
  VELOX_USER_CHECK_NOT_NULL(
      flat, "operation column must be a flat TINYINT vector");
  return flat;
}

// Returns the row_id ROW child. Throws if the row_id column at `channel` is
// not a RowVector with at least 2 children whose first two field types are
// VARCHAR and BIGINT (matching the DV sub-sink's input contract).
const RowVector* asRowIdRowVector(
    const RowVectorPtr& input,
    column_index_t channel) {
  const auto& column = input->childAt(channel);
  VELOX_USER_CHECK_NOT_NULL(column, "row_id column is null");
  const auto* rowVector = column->loadedVector()->as<RowVector>();
  VELOX_USER_CHECK_NOT_NULL(
      rowVector, "row_id column must be a flat ROW vector");
  VELOX_USER_CHECK_GE(
      rowVector->childrenSize(),
      2,
      "row_id must have at least 2 fields (file_path, pos), got {}",
      rowVector->childrenSize());
  VELOX_USER_CHECK(
      rowVector->childAt(0)->type()->isVarchar(),
      "row_id field 0 must be VARCHAR (file_path)");
  VELOX_USER_CHECK(
      rowVector->childAt(1)->type()->isBigint(),
      "row_id field 1 must be BIGINT (pos)");
  return rowVector;
}

// Allocates a buffer big enough for `size` row indices. Returned buffer is
// uninitialized — caller must populate before use.
BufferPtr allocateIndicesBuffer(vector_size_t size, memory::MemoryPool* pool) {
  return AlignedBuffer::allocate<vector_size_t>(size, pool);
}

// Folds the deletion-vector sub-sink's IO/codec counters into the data
// sub-sink's stats. SpillStats are intentionally not merged: the DV sub-sink
// has no spilling path, so the data sub-sink's spillStats is authoritative.
DataSink::Stats mergeSubSinkStats(
    DataSink::Stats dataStats,
    const DataSink::Stats& dvStats) {
  dataStats.numWrittenBytes += dvStats.numWrittenBytes;
  dataStats.numWrittenFiles += dvStats.numWrittenFiles;
  dataStats.writeIOTimeUs += dvStats.writeIOTimeUs;
  dataStats.numCompressedBytes += dvStats.numCompressedBytes;
  dataStats.recodeTimeNs += dvStats.recodeTimeNs;
  dataStats.compressionTimeNs += dvStats.compressionTimeNs;
  return dataStats;
}

} // namespace

IcebergInsertTableHandlePtr IcebergMergeSink::cloneHandleWithKind(
    const IcebergInsertTableHandle& original,
    IcebergInsertTableHandle::WriteKind kind) {
  // Reconstruct the input columns vector from the parent class accessor.
  // HiveInsertTableHandle::inputColumns() returns the wider
  // HiveColumnHandlePtr base type; downcast each back to IcebergColumnHandle
  // (the original construction guarantees this is valid).
  const auto& hiveInputs = original.inputColumns();
  std::vector<IcebergColumnHandlePtr> icebergInputs;
  icebergInputs.reserve(hiveInputs.size());
  for (const auto& col : hiveInputs) {
    icebergInputs.emplace_back(
        checkedPointerCast<const IcebergColumnHandle>(col));
  }
  // Propagate the existing deletion-vector map so the kDeletionVector sub-sink
  // seeds each new DV with the prior DV's positions on an UPDATE/MERGE that
  // re-touches a data file already carrying a DV. The kData sub-sink ignores
  // the map. Preserve the original file-name generator too (the default here
  // would otherwise drop a custom generator).
  return std::make_shared<const IcebergInsertTableHandle>(
      std::move(icebergInputs),
      original.locationHandle(),
      original.storageFormat(),
      original.partitionSpec(),
      original.compressionKind(),
      original.serdeParameters(),
      kind,
      original.existingDeletionVectors(),
      original.fileNameGenerator());
}

RowTypePtr IcebergMergeSink::projectDataInputType(
    const RowTypePtr& inputType,
    const std::vector<column_index_t>& targetColumnChannels,
    const IcebergInsertTableHandle& insertTableHandle) {
  const auto& handleInputColumns = insertTableHandle.inputColumns();
  VELOX_USER_CHECK_EQ(
      targetColumnChannels.size(),
      handleInputColumns.size(),
      "targetColumnChannels ({}) and insertTableHandle.inputColumns() ({}) "
      "must have the same arity",
      targetColumnChannels.size(),
      handleInputColumns.size());
  std::vector<std::string> names;
  std::vector<TypePtr> types;
  names.reserve(targetColumnChannels.size());
  types.reserve(targetColumnChannels.size());
  for (size_t i = 0; i < targetColumnChannels.size(); ++i) {
    const auto channel = targetColumnChannels[i];
    VELOX_USER_CHECK_LT(
        channel,
        inputType->size(),
        "targetColumnChannel {} out of range for inputType of size {}",
        channel,
        inputType->size());
    // Names come from the iceberg insert table handle so the writer's
    // name-based binding always sees iceberg-schema names regardless of
    // upstream naming drift; types still come from the source RowVector
    // schema since that is what we actually receive at runtime.
    VELOX_USER_CHECK_NOT_NULL(
        handleInputColumns[i],
        "insertTableHandle.inputColumns()[{}] is null",
        i);
    names.emplace_back(handleInputColumns[i]->name());
    types.emplace_back(inputType->childAt(channel));
  }
  return ROW(std::move(names), std::move(types));
}

RowTypePtr IcebergMergeSink::makeDeletionVectorInputType() {
  return ROW({"file_path", "pos"}, {VARCHAR(), BIGINT()});
}

IcebergMergeSink::IcebergMergeSink(
    RowTypePtr inputType,
    IcebergInsertTableHandlePtr insertTableHandle,
    const ConnectorQueryCtx* connectorQueryCtx,
    CommitStrategy commitStrategy,
    const std::shared_ptr<const HiveConfig>& hiveConfig,
    const IcebergConfigPtr& icebergConfig,
    std::vector<column_index_t> targetColumnChannels,
    column_index_t operationChannel,
    column_index_t rowIdChannel)
    : inputType_(std::move(inputType)),
      insertTableHandle_(std::move(insertTableHandle)),
      connectorQueryCtx_(connectorQueryCtx),
      targetColumnChannels_(std::move(targetColumnChannels)),
      operationChannel_(operationChannel),
      rowIdChannel_(rowIdChannel),
      dataInputType_(projectDataInputType(
          inputType_,
          targetColumnChannels_,
          *insertTableHandle_)),
      deletionVectorInputType_(makeDeletionVectorInputType()) {
  VELOX_USER_CHECK_NOT_NULL(
      insertTableHandle_,
      "IcebergMergeSink requires a non-null insert table handle.");
  VELOX_USER_CHECK_NOT_NULL(
      inputType_, "IcebergMergeSink requires a non-null input type.");
  VELOX_USER_CHECK_NOT_NULL(
      connectorQueryCtx, "IcebergMergeSink requires a connector query ctx.");
  VELOX_USER_CHECK_LT(
      operationChannel_, inputType_->size(), "operationChannel out of range");
  VELOX_USER_CHECK_LT(
      rowIdChannel_, inputType_->size(), "rowIdChannel out of range");
  VELOX_USER_CHECK(
      inputType_->childAt(operationChannel_)->isTinyint(),
      "operation column must be TINYINT");
  VELOX_USER_CHECK(
      inputType_->childAt(rowIdChannel_)->isRow(),
      "row_id column must be a ROW type");
  VELOX_USER_CHECK_EQ(
      targetColumnChannels_.size(),
      insertTableHandle_->inputColumns().size(),
      "targetColumnChannels size {} must match handle inputColumns size {}",
      targetColumnChannels_.size(),
      insertTableHandle_->inputColumns().size());

  // Build the two narrow handles and instantiate sub-sinks eagerly. Eager
  // construction keeps the lifecycle simple — both sub-sinks always exist
  // and respond to finish / close / abort even if a particular workload
  // produces zero rows for one branch.
  auto dataHandle = cloneHandleWithKind(
      *insertTableHandle_, IcebergInsertTableHandle::WriteKind::kData);
  auto dvHandle = cloneHandleWithKind(
      *insertTableHandle_,
      IcebergInsertTableHandle::WriteKind::kDeletionVector);

  dataSink_ = std::make_unique<IcebergDataSink>(
      dataInputType_,
      std::move(dataHandle),
      connectorQueryCtx_,
      commitStrategy,
      hiveConfig,
      icebergConfig);
  deletionVectorSink_ = std::make_unique<IcebergDeletionVectorSink>(
      deletionVectorInputType_,
      std::move(dvHandle),
      connectorQueryCtx_,
      commitStrategy,
      hiveConfig);
}

RowVectorPtr IcebergMergeSink::makeInsertBatch(
    const RowVectorPtr& input,
    const BufferPtr& insertIndices,
    vector_size_t insertSize) const {
  std::vector<VectorPtr> projectedChildren;
  projectedChildren.reserve(targetColumnChannels_.size());
  for (auto channel : targetColumnChannels_) {
    projectedChildren.push_back(
        BaseVector::wrapInDictionary(
            /*nulls=*/nullptr,
            insertIndices,
            insertSize,
            input->childAt(channel)));
  }
  return std::make_shared<RowVector>(
      connectorQueryCtx_->memoryPool(),
      dataInputType_,
      /*nulls=*/nullptr,
      insertSize,
      std::move(projectedChildren));
}

RowVectorPtr IcebergMergeSink::makeDeleteBatch(
    const RowVectorPtr& input,
    const BufferPtr& deleteIndices,
    vector_size_t deleteSize) const {
  const auto* rowIdRowVector = asRowIdRowVector(input, rowIdChannel_);
  // The DV sink consumes (file_path, pos). The first two children of the
  // row_id ROW carry exactly these. Extra trailing fields (spec_id,
  // partition_data on the full Iceberg row id) are tolerated and ignored.
  auto wrappedFilePath = BaseVector::wrapInDictionary(
      /*nulls=*/nullptr, deleteIndices, deleteSize, rowIdRowVector->childAt(0));
  auto wrappedPos = BaseVector::wrapInDictionary(
      /*nulls=*/nullptr, deleteIndices, deleteSize, rowIdRowVector->childAt(1));
  return std::make_shared<RowVector>(
      connectorQueryCtx_->memoryPool(),
      deletionVectorInputType_,
      /*nulls=*/nullptr,
      deleteSize,
      std::vector<VectorPtr>{
          std::move(wrappedFilePath), std::move(wrappedPos)});
}

void IcebergMergeSink::appendData(RowVectorPtr input) {
  VELOX_USER_CHECK(!finished_, "appendData() called after finish()");
  VELOX_USER_CHECK(!aborted_, "appendData() called after abort()");
  if (input == nullptr || input->size() == 0) {
    return;
  }

  const auto numRows = input->size();
  const auto* operationVector = asOperationFlatVector(input, operationChannel_);

  // First pass: count inserts/deletes and validate operation bytes.
  vector_size_t numInserts = 0;
  vector_size_t numDeletes = 0;
  for (vector_size_t i = 0; i < numRows; ++i) {
    VELOX_USER_CHECK(
        !operationVector->isNullAt(i), "operation byte is null at row {}", i);
    const int8_t op = operationVector->valueAt(i);
    switch (op) {
      case kInsertOperationNumber:
        ++numInserts;
        break;
      case kDeleteOperationNumber:
        ++numDeletes;
        break;
      default:
        VELOX_USER_FAIL(
            "IcebergMergeSink only accepts INSERT (1) and DELETE (2) "
            "operation bytes (UPDATE / DEFAULT must be fanned out by "
            "IcebergMergeProcessor first). Got byte {} at row {}.",
            static_cast<int>(op),
            i);
    }
  }

  // Second pass: bucket row indices by op type.
  auto* pool = connectorQueryCtx_->memoryPool();
  BufferPtr insertIndices;
  BufferPtr deleteIndices;
  vector_size_t* rawInsertIndices = nullptr;
  vector_size_t* rawDeleteIndices = nullptr;
  if (numInserts > 0) {
    insertIndices = allocateIndicesBuffer(numInserts, pool);
    rawInsertIndices = insertIndices->asMutable<vector_size_t>();
  }
  if (numDeletes > 0) {
    deleteIndices = allocateIndicesBuffer(numDeletes, pool);
    rawDeleteIndices = deleteIndices->asMutable<vector_size_t>();
  }
  vector_size_t insertPos = 0;
  vector_size_t deletePos = 0;
  for (vector_size_t i = 0; i < numRows; ++i) {
    const int8_t op = operationVector->valueAt(i);
    if (op == kInsertOperationNumber) {
      rawInsertIndices[insertPos++] = i;
    } else {
      rawDeleteIndices[deletePos++] = i;
    }
  }

  if (numInserts > 0) {
    dataSink_->appendData(makeInsertBatch(input, insertIndices, numInserts));
  }
  if (numDeletes > 0) {
    deletionVectorSink_->appendData(
        makeDeleteBatch(input, deleteIndices, numDeletes));
  }
}

bool IcebergMergeSink::finish() {
  if (finished_) {
    return true;
  }
  // Drive each sub-sink to finish independently, tracking sticky completion
  // so we never re-enter an already-finished sub-sink. Composite reports
  // finished only when both sub-sinks report finished.
  if (!dataSinkFinished_) {
    dataSinkFinished_ = dataSink_->finish();
  }
  if (!deletionVectorSinkFinished_) {
    deletionVectorSinkFinished_ = deletionVectorSink_->finish();
  }
  finished_ = dataSinkFinished_ && deletionVectorSinkFinished_;
  return finished_;
}

std::vector<std::string> IcebergMergeSink::close() {
  // close() is idempotent: drain the sub-sinks once and cache the result.
  // Sub-sink close() is not safe to call twice, so guard re-entry here.
  if (closed_) {
    return commitMessages_;
  }
  closed_ = true;
  // Ensure both sub-sinks have completed their flush state machines. Some
  // sub-sinks (HiveDataSink) may need multiple finish() ticks; close()
  // forces them through to completion. Bound the loop so a sub-sink that
  // never reports finished fails loudly instead of spinning forever.
  constexpr int32_t kMaxFinishIterations = 1'000'000;
  int32_t finishIterations = 0;
  while (!finish()) {
    VELOX_CHECK_LT(
        ++finishIterations,
        kMaxFinishIterations,
        "IcebergMergeSink::close() did not converge after {} finish() "
        "iterations; a sub-sink is stuck mid-flush.",
        kMaxFinishIterations);
  }
  auto dataMessages = dataSink_->close();
  auto deletionMessages = deletionVectorSink_->close();
  // Concatenate commit messages so the coordinator sees a single fragment
  // stream covering both the new data files and the puffin delete files.
  std::vector<std::string> combined;
  combined.reserve(dataMessages.size() + deletionMessages.size());
  combined.insert(
      combined.end(),
      std::make_move_iterator(dataMessages.begin()),
      std::make_move_iterator(dataMessages.end()));
  combined.insert(
      combined.end(),
      std::make_move_iterator(deletionMessages.begin()),
      std::make_move_iterator(deletionMessages.end()));
  commitMessages_ = std::move(combined);
  return commitMessages_;
}

void IcebergMergeSink::abort() {
  if (aborted_) {
    return;
  }
  aborted_ = true;
  // Abort the data sub-sink first because it owns the heavier writer state
  // (open files, memory-pool reservations). The DV sub-sink only holds
  // in-memory roaring bitmaps until finish(); aborting it is essentially
  // free.
  dataSink_->abort();
  deletionVectorSink_->abort();
}

DataSink::Stats IcebergMergeSink::stats() const {
  return mergeSubSinkStats(dataSink_->stats(), deletionVectorSink_->stats());
}

} // namespace facebook::velox::connector::hive::iceberg
