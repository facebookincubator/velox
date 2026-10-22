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
  paimonTableHandle_->validateReadProfile();
  for (const auto& [_, handle] : assignments) {
    paimonTableHandle_->validateColumn(
        *static_cast<const FileColumnHandle*>(handle.get()));
  }
  VELOX_USER_CHECK(
      !paimonConfig->ignoreMissingFiles(connectorQueryCtx->sessionProperties()),
      "Paimon same-schema append profile does not support ignore_missing_files; "
      "missing snapshot files must fail instead of returning incomplete results");
}

void PaimonDataSource::addSplit(std::shared_ptr<ConnectorSplit> split) {
  auto paimonSplit = checkedPointerCast<PaimonConnectorSplit>(split);
  paimonTableHandle_->validateSplit(*paimonSplit);

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
      VELOX_CHECK_NOT_NULL(sourceField);
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
