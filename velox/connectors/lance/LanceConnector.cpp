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

#include "velox/vector/arrow/Bridge.h"

namespace facebook::velox::connector::lance {

namespace {

// Runs a lance-c call and rethrows any ::lance::Error as a Velox error.
// lance-c reports failures through a thread-local error slot whose message
// string is heap allocated; ::lance::check_error() (used throughout
// lance.hpp) frees it before throwing, so routing everything through here
// avoids leaking it.
template <typename Func>
auto callLance(const char* context, Func&& func) -> decltype(func()) {
  try {
    return func();
  } catch (const ::lance::Error& error) {
    VELOX_FAIL(
        "{}: lance error {}: {}",
        context,
        static_cast<int>(error.code),
        error.what());
  }
}

} // namespace

LanceConnector::LanceConnector(
    const std::string& id,
    std::shared_ptr<const config::ConfigBase> config,
    folly::Executor* /*executor*/)
    : Connector(id, std::move(config)) {}

std::unique_ptr<DataSource> LanceConnector::createDataSource(
    const RowTypePtr& outputType,
    const ConnectorTableHandlePtr& tableHandle,
    const connector::ColumnHandleMap& columnHandles,
    ConnectorQueryCtx* connectorQueryCtx) {
  return std::make_unique<LanceDataSource>(
      outputType, tableHandle, columnHandles, connectorQueryCtx);
}

std::unique_ptr<DataSink> LanceConnector::createDataSink(
    RowTypePtr /*inputType*/,
    ConnectorInsertTableHandlePtr /*connectorInsertTableHandle*/,
    ConnectorQueryCtx* /*connectorQueryCtx*/,
    CommitStrategy /*commitStrategy*/) {
  VELOX_NYI("LanceConnector does not support data sink.");
}

std::shared_ptr<Connector> LanceConnectorFactory::newConnector(
    const std::string& id,
    std::shared_ptr<const config::ConfigBase> config,
    folly::Executor* ioExecutor,
    folly::Executor* /*cpuExecutor*/) {
  return std::make_shared<LanceConnector>(id, std::move(config), ioExecutor);
}

LanceDataSource::LanceDataSource(
    const RowTypePtr& outputType,
    const connector::ConnectorTableHandlePtr& tableHandle,
    const connector::ColumnHandleMap& columnHandles,
    ConnectorQueryCtx* connectorQueryCtx)
    : outputType_(outputType), pool_(connectorQueryCtx->memoryPool()) {
  auto lanceTableHandle =
      std::dynamic_pointer_cast<const LanceTableHandle>(tableHandle);
  VELOX_CHECK_NOT_NULL(
      lanceTableHandle, "TableHandle must be an instance of LanceTableHandle.");

  // The version, storage options and pushed filter are resolved once during
  // planning and apply to every split, so read them here rather than in
  // addSplit().
  datasetVersion_ = lanceTableHandle->datasetVersion();
  filter_ = lanceTableHandle->filter();

  const auto& storageOptions = lanceTableHandle->storageOptions();
  storageOptions_.reserve(storageOptions.size());
  for (const auto& [key, value] : storageOptions) {
    storageOptions_.emplace_back(key, value);
  }

  // Build the list of column names to project from the column handles.
  columnNames_.reserve(outputType->size());
  for (const auto& outputName : outputType->names()) {
    auto it = columnHandles.find(outputName);
    VELOX_USER_CHECK(
        it != columnHandles.end(),
        "Missing ColumnHandle for output column: {}",
        outputName);

    auto handle =
        std::dynamic_pointer_cast<const LanceColumnHandle>(it->second);
    VELOX_CHECK_NOT_NULL(
        handle,
        "ColumnHandle must be an instance of LanceColumnHandle: {}",
        it->second->name());

    columnNames_.push_back(handle->name());
  }
}

LanceDataSource::~LanceDataSource() {
  closeScannerAndDataset();
}

void LanceDataSource::closeScannerAndDataset() {
  // Order matters: the scanner borrows the dataset handle.
  scanner_.reset();
  dataset_.reset();
}

void LanceDataSource::addSplit(std::shared_ptr<ConnectorSplit> split) {
  VELOX_CHECK(
      splitProcessed_,
      "Previous split has not been processed yet."
      " Call next() to process the split.");

  auto lanceSplit = std::dynamic_pointer_cast<LanceConnectorSplit>(split);
  VELOX_CHECK_NOT_NULL(lanceSplit, "Wrong type of split for LanceDataSource.");

  // Clean up any prior state.
  closeScannerAndDataset();

  const auto& path = lanceSplit->datasetPath;

  dataset_ = callLance(
      fmt::format(
          "Failed to open Lance dataset at path {} version {}",
          path,
          datasetVersion_)
          .c_str(),
      [&] {
        return ::lance::Dataset::open(path, storageOptions_, datasetVersion_);
      });

  // Build the NULL-terminated column projection array. Lance returns columns
  // in the order requested here, which projectOutputColumns() verifies.
  std::vector<const char*> columnPointers;
  columnPointers.reserve(columnNames_.size() + 1);
  for (const auto& column : columnNames_) {
    columnPointers.push_back(column.c_str());
  }
  columnPointers.push_back(nullptr);

  // lance.hpp's Dataset::scan() cannot express column projection, so create
  // the scanner through the C entry point and adopt it into the RAII wrapper.
  auto* rawScanner = lance_scanner_new(
      dataset_->c_handle(),
      columnPointers.data(),
      filter_.empty() ? nullptr : filter_.c_str());
  if (rawScanner == nullptr) {
    // A rejected predicate is the most likely cause, so name it here.
    callLance(
        fmt::format(
            "Failed to create scanner at path {} with filter {}",
            path,
            filter_.empty() ? "<none>" : filter_)
            .c_str(),
        [] { ::lance::check_error(); });
    VELOX_FAIL(
        "Failed to create Lance scanner at path {} with filter {}",
        path,
        filter_.empty() ? "<none>" : filter_);
  }
  scanner_.emplace(rawScanner);

  // If the split specifies fragment IDs, restrict the scan.
  if (!lanceSplit->fragmentIds.empty()) {
    callLance("Failed to set fragment IDs", [&] {
      scanner_->fragment_ids(lanceSplit->fragmentIds);
    });
  }

  splitProcessed_ = false;
  batchSizeApplied_ = false;
}

RowVectorPtr LanceDataSource::projectOutputColumns(const RowVectorPtr& vector) {
  const auto& inputType = vector->type()->asRow();

  std::vector<VectorPtr> children;
  children.reserve(outputType_->size());

  // Lance projects columns in the order they were requested, which is
  // outputType_ order. Look them up by name anyway so that a change in that
  // behaviour surfaces as an error instead of silently mislabelling columns.
  for (size_t i = 0; i < outputType_->size(); ++i) {
    const auto& name = outputType_->nameOf(i);
    const auto channel = inputType.getChildIdxIfExists(name);
    VELOX_CHECK(
        channel.has_value(),
        "Lance batch is missing projected column: {}",
        name);

    const auto& child = vector->childAt(channel.value());
    VELOX_CHECK(
        child->type()->equivalent(*outputType_->childAt(i)),
        "Type mismatch for Lance column {}: expected {}, got {}",
        name,
        outputType_->childAt(i)->toString(),
        child->type()->toString());

    children.push_back(child);
  }

  return std::make_shared<RowVector>(
      pool_, outputType_, BufferPtr(), vector->size(), std::move(children));
}

std::optional<RowVectorPtr> LanceDataSource::next(
    uint64_t size,
    velox::ContinueFuture& /*future*/) {
  if (splitProcessed_) {
    return nullptr;
  }

  VELOX_CHECK(scanner_.has_value(), "No active scanner.");

  // Honour the batch size the caller asked for. Lance only accepts scanner
  // settings before iteration begins, so apply it once per split rather than
  // on every call.
  if (!batchSizeApplied_) {
    callLance("Failed to set batch size", [&] {
      scanner_->batch_size(static_cast<int64_t>(size));
    });
    batchSizeApplied_ = true;
  }

  LanceBatch* rawBatch = nullptr;
  const int32_t rc = lance_scanner_next(scanner_->c_handle(), &rawBatch);

  if (rc == 1) {
    // Stream exhausted.
    splitProcessed_ = true;
    closeScannerAndDataset();
    return nullptr;
  }

  if (rc < 0) {
    callLance(
        "Error reading from Lance scanner", [] { ::lance::check_error(); });
    VELOX_FAIL("Error reading from Lance scanner.");
  }

  // Adopt the batch so it is freed even if the conversion below throws.
  ::lance::Batch batch(rawBatch);

  ArrowArray arrowArray{};
  ArrowSchema arrowSchema{};
  callLance("Failed to convert Lance batch to Arrow", [&] {
    batch.to_arrow(&arrowArray, &arrowSchema);
  });

  // Import Arrow data into Velox.
  auto result = importFromArrowAsOwner(arrowSchema, arrowArray, pool_);
  auto rowVector = std::dynamic_pointer_cast<RowVector>(result);
  VELOX_CHECK_NOT_NULL(rowVector, "Arrow import did not produce a RowVector.");

  completedRows_ += rowVector->size();
  completedBytes_ += rowVector->retainedSize();

  return projectOutputColumns(rowVector);
}

} // namespace facebook::velox::connector::lance
