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

#include <array>

#include "velox/vector/arrow/Bridge.h"

namespace facebook::velox::connector::lance {

namespace {

// Name of the distance column Lance appends to every batch of a nearest
// scan. It is synthesized by the scan, not stored in the dataset, so it must
// never be part of the projection handed to lance_scanner_new.
constexpr const char* kDistanceColumnName = "_distance";

// Maps the metric names the coordinator produces (see the Java
// LanceVectorSearch constants) to lance-c metric types. Hamming exists in
// lance-c but is never produced by the coordinator, so it is rejected too.
LanceMetricType toLanceMetric(const std::string& metric) {
  if (metric == "l2") {
    return LANCE_METRIC_L2;
  }
  if (metric == "cosine") {
    return LANCE_METRIC_COSINE;
  }
  if (metric == "dot") {
    return LANCE_METRIC_DOT;
  }
  VELOX_USER_FAIL("Unsupported Lance vector distance metric: {}", metric);
}

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

// Converts a hex digit character to its numeric value (0-15), or -1 if the
// character is not a valid hex digit.
int hexDigit(char c) {
  if (c >= '0' && c <= '9') {
    return c - '0';
  }
  if (c >= 'a' && c <= 'f') {
    return c - 'a' + 10;
  }
  if (c >= 'A' && c <= 'F') {
    return c - 'A' + 10;
  }
  return -1;
}

// Parses canonical UUID strings (8-4-4-4-12 hex form, e.g.
// "550e8400-e29b-41d4-a716-446655440000") into 16-byte arrays suitable for
// lance-c Scanner::index_segments.
std::vector<std::array<uint8_t, 16>> parseSegmentUuids(
    const std::vector<std::string>& uuidStrings) {
  std::vector<std::array<uint8_t, 16>> result;
  result.reserve(uuidStrings.size());
  for (const auto& s : uuidStrings) {
    VELOX_USER_CHECK(
        s.size() == 36, "UUID must be 36 characters (8-4-4-4-12): {}", s);
    VELOX_USER_CHECK(
        s[8] == '-' && s[13] == '-' && s[18] == '-' && s[23] == '-',
        "UUID must have dashes at positions 8, 13, 18, 23: {}",
        s);
    std::array<uint8_t, 16> uuid{};
    size_t byteIdx = 0;
    for (size_t i = 0; i < 36; ++i) {
      if (s[i] == '-') {
        continue;
      }
      const int hi = hexDigit(s[i]);
      VELOX_USER_CHECK(hi >= 0, "Non-hex character in UUID: {}", s);
      ++i;
      const int lo = hexDigit(s[i]);
      VELOX_USER_CHECK(lo >= 0, "Non-hex character in UUID: {}", s);
      uuid[byteIdx++] = static_cast<uint8_t>((hi << 4) | lo);
    }
    result.push_back(uuid);
  }
  return result;
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
  vectorSearch_ = lanceTableHandle->vectorSearch();

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

void LanceDataSource::configureVectorSearch() {
  const auto& search = vectorSearch_.value();
  // Resolve the metric first so an unsupported name fails before any
  // scanner state is touched.
  const LanceMetricType metric = toLanceMetric(search.metric);
  callLance("Failed to configure Lance vector search", [&] {
    scanner_->nearest(
        search.column,
        search.queryVector.data(),
        search.queryVector.size(),
        search.k);
    scanner_->metric(metric);
    // Prefer the pre-built vector index when one exists; Lance falls back
    // to a flat search otherwise. Mirrors the Java setUseIndex(true).
    scanner_->use_index(true);
    if (search.nprobes.has_value()) {
      scanner_->nprobes(search.nprobes.value());
    }
    if (search.ef.has_value()) {
      scanner_->ef(search.ef.value());
    }
    if (search.refineFactor.has_value()) {
      scanner_->refine_factor(search.refineFactor.value());
    }
  });
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
  // in the order requested here, which projectOutputColumns() verifies. The
  // _distance column is synthesized by a nearest scan rather than stored, so
  // it is never requested; if nothing else is selected, pass no projection
  // (Lance then returns all columns) rather than an empty one.
  std::vector<const char*> columnPointers;
  columnPointers.reserve(columnNames_.size() + 1);
  for (const auto& column : columnNames_) {
    if (column != kDistanceColumnName) {
      columnPointers.push_back(column.c_str());
    }
  }
  const bool hasProjection = !columnPointers.empty();
  columnPointers.push_back(nullptr);

  // lance.hpp's Dataset::scan() cannot express column projection, so create
  // the scanner through the C entry point and adopt it into the RAII wrapper.
  auto* rawScanner = lance_scanner_new(
      dataset_->c_handle(),
      hasProjection ? columnPointers.data() : nullptr,
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

  // Route by split type. INDEXED splits (segmentUuids non-empty) configure
  // index_segments after nearest; FLAT splits (flatSearch=true) disable the
  // index after configureVectorSearch; plain fragment-scoped splits keep the
  // existing prefilter path.
  if (!lanceSplit->segmentUuids.empty()) {
    // INDEXED split: search only these index segments. Lance filters the
    // loaded index deltas to the requested UUIDs at plan time, so this split
    // probes exactly the specified segments' indexes; rows outside the
    // segments are excluded (the coordinator emits a separate flat split for
    // them).
    VELOX_USER_CHECK(
        vectorSearch_.has_value(),
        "segmentUuids is only valid for a vector search split");
    VELOX_USER_CHECK(
        lanceSplit->fragmentIds.empty(),
        "A split must not set both segmentUuids and fragmentIds");
    VELOX_USER_CHECK(
        !lanceSplit->flatSearch,
        "A split must not set both segmentUuids and flatSearch");
  } else if (!lanceSplit->fragmentIds.empty()) {
    // Fragment-scoped scan. Both the standard fragment path and the
    // flat-search path need prefilter enabled: Rust's nearest() rejects a
    // fragment scan unless prefilter is on (scanner.rs:1432-1436), and the
    // fragment subset becomes a pre-ranking row mask that restricts results
    // to rows within the split's fragments.
    callLance("Failed to set fragment IDs", [&] {
      scanner_->fragment_ids(lanceSplit->fragmentIds);
      if (vectorSearch_.has_value()) {
        scanner_->prefilter(true);
      }
    });
  }

  if (vectorSearch_.has_value()) {
    configureVectorSearch();
  }

  // INDEXED: set index_segments after configureVectorSearch because lance-c
  // validates that nearest was configured before accepting index_segments.
  if (!lanceSplit->segmentUuids.empty()) {
    auto uuids = parseSegmentUuids(lanceSplit->segmentUuids);
    callLance("Failed to set index segments", [&] {
      scanner_->index_segments(uuids);
    });
  }

  // FLAT split: brute-force KNN over the split's fragments (rows not covered
  // by any index segment). The index is disabled so Lance does not pointlessly
  // probe it. Must come after configureVectorSearch to override the
  // use_index(true) it sets by default.
  if (lanceSplit->flatSearch) {
    VELOX_USER_CHECK(
        vectorSearch_.has_value(),
        "flatSearch is only valid for a vector search split");
    callLance("Failed to disable index", [&] { scanner_->use_index(false); });
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
