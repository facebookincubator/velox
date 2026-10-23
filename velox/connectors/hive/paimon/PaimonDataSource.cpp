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
#include "velox/connectors/hive/paimon/PaimonDataSource.h"

#include "velox/common/Casts.h"
#include "velox/connectors/hive/paimon/PaimonConnectorSplit.h"
#include "velox/connectors/hive/paimon/PaimonSplitReader.h"
#include "velox/connectors/hive/paimon/PaimonTableHandle.h"

namespace facebook::velox::connector::hive::paimon {

PaimonDataSource::PaimonDataSource(
    const RowTypePtr& outputType,
    const std::shared_ptr<const PaimonTableHandle>& tableHandle,
    const ColumnHandleMap& assignments,
    FileHandleFactory* fileHandleFactory,
    folly::Executor* ioExecutor,
    const ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<PaimonConfig>& paimonConfig)
    : FileDataSource(
          outputType,
          tableHandle,
          assignments,
          fileHandleFactory,
          ioExecutor,
          connectorQueryCtx,
          paimonConfig),
      paimonTableHandle_(tableHandle) {
  // TODO: Add the planned PK, DV, DE and thin-file readers when their row and
  // layout semantics are implemented. These are reader limits, not schema
  // validity rules; catalog and snapshot planning remain with the caller.
  VELOX_USER_CHECK(
      tableHandle->tableType() == PaimonTableType::kAppendOnly,
      "Paimon same-schema append reader does not support primary-key tables; "
      "primary-key reading is not implemented yet");
  const auto& options = tableHandle->options();
  for (const auto* key :
       {"deletion-vectors.enabled",
        "data-evolution.enabled",
        "data-file.thin-mode"}) {
    const auto it = options.find(key);
    VELOX_USER_CHECK(
        it != options.end() && it->second == "false",
        "Paimon same-schema append reader requires normalized '{}'='false'; "
        "the corresponding read semantics are not implemented yet",
        key);
  }
  const auto tracking = options.find("row-tracking.enabled");
  VELOX_USER_CHECK(
      tracking != options.end() &&
          (tracking->second == "true" || tracking->second == "false"),
      "Paimon reader requires normalized boolean 'row-tracking.enabled'");

  // The input already uses Velox types. Do not introduce a parallel logical
  // type model to describe capabilities that this reader does not yet expose.
  for (const auto& type : tableHandle->rowType()->children()) {
    VELOX_USER_CHECK(
        *type == *BOOLEAN() || *type == *INTEGER() || *type == *BIGINT() ||
            *type == *DOUBLE() || *type == *VARCHAR(),
        "Unsupported Paimon same-schema append type '{}'; the current reader "
        "supports BOOLEAN, INTEGER, BIGINT, DOUBLE and VARCHAR",
        type->toString());
  }
  // FileDataSource has resolved assignments and expression dependencies. Check
  // that actual reader inputs describe the fixed table, without rebinding
  // query columns by ID or revalidating column roles and expression trees.
  for (column_index_t i = 0; i < readerOutputType_->size(); ++i) {
    const auto& name = readerOutputType_->nameOf(i);
    const auto index = tableHandle->rowType()->getChildIdxIfExists(name);
    VELOX_USER_CHECK(
        index.has_value(),
        "Unknown Paimon scan column '{}' in fixed table",
        name);
    VELOX_USER_CHECK(
        *readerOutputType_->childAt(i) ==
            *tableHandle->rowType()->childAt(*index),
        "Paimon scan column '{}' type does not match fixed table",
        name);
  }
  VELOX_USER_CHECK(
      !paimonConfig->ignoreMissingFiles(connectorQueryCtx->sessionProperties()),
      "Paimon reader does not support ignore_missing_files; "
      "missing snapshot files must fail instead of returning incomplete results");
}

void PaimonDataSource::addSplit(std::shared_ptr<ConnectorSplit> split) {
  auto paimonSplit = checkedPointerCast<PaimonConnectorSplit>(split);
  VELOX_USER_CHECK_EQ(
      paimonSplit->connectorId,
      paimonTableHandle_->connectorId(),
      "Paimon split connector does not match table handle");
  VELOX_USER_CHECK_EQ(
      paimonSplit->tableType(),
      paimonTableHandle_->tableType(),
      "Paimon split table type does not match table handle");
  VELOX_USER_CHECK(
      paimonSplit->rawConvertible(),
      "Paimon same-schema append reader requires rawConvertible=true; "
      "merge-on-read is not implemented yet");
  for (const auto& file : paimonSplit->dataFiles()) {
    VELOX_USER_CHECK(
        !file.deletionFile,
        "Paimon reader does not support deletion vectors; "
        "position-based deletion filtering is not implemented yet");
    VELOX_USER_CHECK_EQ(
        file.type,
        PaimonDataFile::Type::kData,
        "Paimon reader requires data files; changelog reading is not implemented yet");
  }
  const auto& partitions = paimonSplit->partitionKeys();
  VELOX_USER_CHECK_EQ(
      partitions.size(),
      paimonTableHandle_->partitionKeys().size(),
      "Paimon split partition names do not match table handle");
  for (const auto& name : paimonTableHandle_->partitionKeys()) {
    VELOX_USER_CHECK(
        partitions.count(name), "Missing Paimon split partition '{}'", name);
  }

  paimonSplit_ = std::move(paimonSplit);

  // Create a FileConnectorSplit for the first data file and delegate to
  // FileDataSource::addSplit(), which calls createSplitReader() to create
  // a PaimonSplitReader that handles all files internally.
  const auto& firstFile = paimonSplit_->dataFiles().front();
  auto firstFileSplit = std::make_shared<FileConnectorSplit>(
      paimonSplit_->connectorId,
      firstFile.path,
      paimonSplit_->fileFormat(),
      /*_start=*/0,
      /*_length=*/std::numeric_limits<uint64_t>::max(),
      /*splitWeight=*/0,
      /*cacheable=*/true,
      /*_properties=*/std::nullopt,
      paimonSplit_->partitionKeys());

  FileDataSource::addSplit(std::move(firstFileSplit));
}

std::unique_ptr<FileSplitReader> PaimonDataSource::createSplitReader() {
  return std::make_unique<PaimonSplitReader>(
      split_,
      paimonSplit_,
      paimonTableHandle_,
      &partitionKeys_,
      connectorQueryCtx_,
      fileConfig_,
      readerOutputType_,
      dataIoStats_,
      metadataIoStats_,
      ioStats_,
      fileHandleFactory_,
      ioExecutor_,
      fileScanState_->scanSpec);
}

void PaimonDataSource::setFromDataSource(
    std::unique_ptr<DataSource> sourceUnique) {
  auto* source = checkedPointerCast<PaimonDataSource>(sourceUnique.get());
  auto* sourceReader =
      checkedPointerCast<PaimonSplitReader>(source->splitReader_.get());
  // Common adaptation skips constant columns. A prior split's partition
  // constant must not prevent its current filters from reaching the donor.
  for (const auto& [name, column] : partitionKeys_) {
    const auto* field = scanSpec_->childByName(name);
    if (field && field->filter()) {
      auto* sourceField = source->scanSpec_->childByName(name);
      sourceField->setFilter(field->filter()->clone());
    }
  }
  paimonSplit_ = std::move(source->paimonSplit_);
  FileDataSource::setFromDataSource(std::move(sourceUnique));
  // Files are opened lazily after takeover, when the previous owner of the
  // borrowed runtimeStats pointer has already been destroyed.
  sourceReader->setRuntimeStats(runtimeStats_);
}

} // namespace facebook::velox::connector::hive::paimon
