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
#include "velox/dwio/common/ReaderFactory.h"

namespace facebook::velox::connector::hive::paimon {
namespace {

const std::shared_ptr<const PaimonTableHandle>& validateTableInput(
    const RowTypePtr& outputType,
    const std::shared_ptr<const PaimonTableHandle>& handle,
    const ColumnHandleMap& assignments,
    const ConnectorQueryCtx* context,
    const std::shared_ptr<PaimonConfig>& config) {
  VELOX_USER_CHECK_NOT_NULL(handle, "Missing Paimon table handle");
  VELOX_CHECK_NOT_NULL(context);
  VELOX_CHECK_NOT_NULL(config);
  handle->validateQuery(outputType, assignments);
  VELOX_USER_CHECK(
      !config->ignoreMissingFiles(context->sessionProperties()),
      "Paimon same-schema append profile does not support ignore_missing_files");
  VELOX_USER_CHECK(
      !config->isFileColumnNamesReadAsLowerCase(context->sessionProperties()),
      "Paimon same-schema append profile does not support lower-case file mapping");
  return handle;
}

} // namespace

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
          validateTableInput(
              outputType,
              tableHandle,
              assignments,
              connectorQueryCtx,
              paimonConfig),
          assignments,
          fileHandleFactory,
          ioExecutor,
          connectorQueryCtx,
          paimonConfig),
      paimonTableHandle_(tableHandle) {}

void PaimonDataSource::addSplit(std::shared_ptr<ConnectorSplit> split) {
  auto paimonSplit = checkedPointerCast<PaimonConnectorSplit>(split);
  paimonTableHandle_->validateSplit(*paimonSplit);
  VELOX_USER_CHECK(
      dwio::common::hasReaderFactory(paimonSplit->fileFormat()),
      "No Paimon reader factory for file format {}",
      dwio::common::FileFormatName::toName(paimonSplit->fileFormat()));

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
  VELOX_USER_CHECK(
      paimonTableHandle_->hasSameTableDefinition(*source->paimonTableHandle_),
      "Paimon takeover must preserve the fixed table definition");
  VELOX_USER_CHECK(
      paimonTableHandle_->hasSameScanConditions(*source->paimonTableHandle_) &&
          *outputType() == *source->outputType() &&
          *readerOutputType_ == *source->readerOutputType_ &&
          partitionKeys_.size() == source->partitionKeys_.size(),
      "Paimon takeover must preserve scan conditions and query columns");
  for (const auto& [name, column] : partitionKeys_) {
    VELOX_USER_CHECK(
        source->partitionKeys_.count(name),
        "Paimon takeover must preserve partition column handles");
  }
  VELOX_USER_CHECK_NULL(split_, "Paimon takeover requires an idle recipient");
  VELOX_USER_CHECK_NOT_NULL(
      source->paimonSplit_, "Paimon takeover requires a prepared split");
  VELOX_USER_CHECK_NOT_NULL(
      source->split_, "Paimon takeover requires a prepared split");
  VELOX_USER_CHECK_NOT_NULL(
      source->splitReader_, "Paimon takeover requires a prepared reader");
  VELOX_USER_CHECK(
      !checkedPointerCast<PaimonSplitReader>(source->splitReader_.get())
           ->startedReading(),
      "Paimon takeover does not support an already-read donor");
  paimonTableHandle_->validateSplit(*source->paimonSplit_);
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
  checkedPointerCast<PaimonSplitReader>(splitReader_.get())
      ->setRuntimeStats(runtimeStats_);
}

void PaimonDataSource::addDynamicFilter(
    column_index_t outputChannel,
    const std::shared_ptr<common::Filter>& filter) {
  VELOX_USER_CHECK_LT(
      outputChannel,
      outputType()->size(),
      "Invalid Paimon dynamic filter channel");
  VELOX_USER_CHECK_NOT_NULL(filter, "Missing Paimon dynamic filter");
  paimonTableHandle_->validateDynamicFilter(
      outputType()->childAt(outputChannel), *filter);
  const auto* existing = scanSpec_->getChildByChannel(outputChannel).filter();
  if (existing) {
    FileDataSource::addDynamicFilter(
        outputChannel, existing->mergeWith(filter.get()));
    return;
  }
  FileDataSource::addDynamicFilter(outputChannel, filter);
}

} // namespace facebook::velox::connector::hive::paimon
