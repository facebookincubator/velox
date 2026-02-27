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

#include "velox/connectors/lance/LanceConnector.h"

#include "velox/connectors/lance/lance_ffi.h"
#include "velox/vector/arrow/Bridge.h"

namespace facebook::velox::connector::lance {

LanceConnector::LanceConnector(
    const std::string& id,
    std::shared_ptr<const config::ConfigBase> config,
    folly::Executor* /*executor*/)
    : Connector(id, std::move(config)) {}

LanceDataSource::LanceDataSource(
    const RowTypePtr& outputType,
    const connector::ConnectorTableHandlePtr& tableHandle,
    const connector::ColumnHandleMap& columnHandles,
    ConnectorQueryCtx* connectorQueryCtx)
    : outputType_(outputType), pool_(connectorQueryCtx->memoryPool()) {
  auto lanceTableHandle =
      std::dynamic_pointer_cast<const LanceTableHandle>(tableHandle);
  VELOX_CHECK_NOT_NULL(
      lanceTableHandle,
      "TableHandle must be an instance of LanceTableHandle");

  // Build the list of column names to project from the column handles.
  columnNames_.reserve(outputType->size());
  for (const auto& outputName : outputType->names()) {
    auto it = columnHandles.find(outputName);
    VELOX_CHECK(
        it != columnHandles.end(),
        "ColumnHandle is missing for output column '{}'",
        outputName);

    auto handle =
        std::dynamic_pointer_cast<const LanceColumnHandle>(it->second);
    VELOX_CHECK_NOT_NULL(
        handle,
        "ColumnHandle must be an instance of LanceColumnHandle for '{}'",
        it->second->name());

    columnNames_.push_back(handle->name());
  }
}

LanceDataSource::~LanceDataSource() {
  closeStreamAndDataset();
}

void LanceDataSource::closeStreamAndDataset() {
  if (stream_) {
    lance_close_stream(stream_);
    stream_ = nullptr;
  }
  if (dataset_) {
    lance_close_dataset(dataset_);
    dataset_ = nullptr;
  }
}

void LanceDataSource::addSplit(std::shared_ptr<ConnectorSplit> split) {
  VELOX_CHECK(
      splitProcessed_,
      "Previous split has not been processed yet. Call next() to process the split.");

  auto lanceSplit = std::dynamic_pointer_cast<LanceConnectorSplit>(split);
  VELOX_CHECK(lanceSplit, "Wrong type of split for LanceDataSource.");

  // Clean up any prior state.
  closeStreamAndDataset();

  // Open the dataset.
  dataset_ = lance_open_dataset(lanceSplit->datasetPath.c_str());
  VELOX_CHECK_NOT_NULL(
      dataset_,
      "Failed to open Lance dataset at '{}': {}",
      lanceSplit->datasetPath,
      lance_last_error_message());

  // Build column projection pointers for the FFI call.
  std::vector<const char*> colPtrs;
  colPtrs.reserve(columnNames_.size());
  for (const auto& col : columnNames_) {
    colPtrs.push_back(col.c_str());
  }

  // Create the scan stream with column projection.
  stream_ = lance_create_dataset_stream_ir(
      dataset_,
      colPtrs.data(),
      colPtrs.size(),
      /*filter_ir=*/nullptr,
      /*filter_ir_len=*/0,
      /*limit=*/-1,
      /*offset=*/0);
  VELOX_CHECK_NOT_NULL(
      stream_,
      "Failed to create scan stream for Lance dataset '{}': {}",
      lanceSplit->datasetPath,
      lance_last_error_message());

  splitProcessed_ = false;
}

RowVectorPtr LanceDataSource::projectOutputColumns(RowVectorPtr vector) {
  // The Arrow import gives us columns in the order we requested them, which
  // matches the order of columnNames_. Build a RowVector that maps to the
  // output type.
  auto inputType = std::dynamic_pointer_cast<const RowType>(vector->type());
  VELOX_CHECK_NOT_NULL(inputType);

  std::vector<VectorPtr> children;
  children.reserve(outputType_->size());

  for (size_t i = 0; i < outputType_->size(); ++i) {
    // Columns are returned in the same order as columnNames_, which matches
    // the outputType_ order by construction.
    children.push_back(vector->childAt(i));
  }

  return std::make_shared<RowVector>(
      pool_,
      outputType_,
      BufferPtr(),
      vector->size(),
      std::move(children));
}

std::optional<RowVectorPtr> LanceDataSource::next(
    uint64_t /*size*/,
    velox::ContinueFuture& /*future*/) {
  if (splitProcessed_) {
    return nullptr;
  }

  VELOX_CHECK_NOT_NULL(stream_, "No active scan stream.");

  void* batch = nullptr;
  int32_t rc = lance_stream_next(stream_, &batch);

  if (rc == 1) {
    // Stream exhausted.
    splitProcessed_ = true;
    closeStreamAndDataset();
    return nullptr;
  }

  VELOX_CHECK_GE(
      rc,
      0,
      "Error reading from Lance stream: {}",
      lance_last_error_message());

  // Convert the batch to Arrow C Data Interface structs.
  ArrowArray arrowArray;
  ArrowSchema arrowSchema;
  int32_t convertRc = lance_batch_to_arrow(batch, &arrowArray, &arrowSchema);
  lance_free_batch(batch);

  VELOX_CHECK_EQ(
      convertRc,
      0,
      "Failed to convert Lance batch to Arrow: {}",
      lance_last_error_message());

  // Import Arrow data into Velox.
  auto result = importFromArrowAsOwner(arrowSchema, arrowArray, pool_);
  auto rowVector = std::dynamic_pointer_cast<RowVector>(result);
  VELOX_CHECK_NOT_NULL(rowVector, "Arrow import did not produce a RowVector");

  completedRows_ += rowVector->size();
  completedBytes_ += rowVector->retainedSize();

  return projectOutputColumns(rowVector);
}

} // namespace facebook::velox::connector::lance
