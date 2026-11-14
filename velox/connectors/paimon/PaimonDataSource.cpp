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
#include "velox/connectors/paimon/PaimonDataSource.h"

#include "velox/common/Casts.h"
#include "velox/connectors/paimon/PaimonConnectorSplit.h"
#include "velox/connectors/paimon/PaimonMergeSplitReader.h"
#include "velox/connectors/paimon/PaimonSplitReader.h"
#include "velox/connectors/paimon/PaimonTableHandle.h"

namespace facebook::velox::connector::paimon {

PaimonDataSource::PaimonDataSource(
    const RowTypePtr& outputType,
    const std::shared_ptr<const PaimonTableHandle>& tableHandle,
    const ColumnHandleMap& assignments,
    FileHandleFactory* fileHandleFactory,
    folly::Executor* ioExecutor,
    const ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<PaimonConfig>& paimonConfig)
    : connector::FileDataSource(
          outputType,
          tableHandle,
          assignments,
          fileHandleFactory,
          ioExecutor,
          connectorQueryCtx,
          paimonConfig),
      paimonTableHandle_(tableHandle) {}

void PaimonDataSource::addSplit(std::shared_ptr<ConnectorSplit> split) {
  auto paimonSplit = checkedPointerCast<PaimonConnectorSplit>(split);
  VELOX_USER_CHECK_EQ(
      paimonSplit->tableType(),
      paimonTableHandle_->tableType(),
      "Paimon split table type does not match table handle");
  // Changelog files serve streaming reads, which return row kinds.
  for (const auto& file : paimonSplit->dataFiles()) {
    VELOX_USER_CHECK_EQ(
        file.type,
        PaimonDataFile::Type::kData,
        "Paimon reader does not support changelog files: {}",
        file.path);
  }

  if (!paimonSplit->rawConvertible()) {
    VELOX_CHECK_EQ(
        paimonSplit->tableType(),
        PaimonTableType::kPrimaryKey,
        "Only primary-key Paimon tables can require merge-on-read");
    VELOX_USER_CHECK(
        !paimonTableHandle_->trimmedPrimaryKeys().empty(),
        "Paimon merge-on-read requires the primary keys of the table: {}",
        paimonTableHandle_->tableName());
    const auto mergeEngine = paimonTableHandle_->mergeOptions().mergeEngine;
    if (mergeEngine == PaimonMergeEngine::kPartialUpdate ||
        mergeEngine == PaimonMergeEngine::kAggregation) {
      VELOX_NYI(
          "Paimon merge-on-read does not support the merge engine: {}",
          mergeEngine);
    }
  }
  paimonSplit_ = std::move(paimonSplit);

  // Use the first file to enter the FileDataSource scan pipeline;
  // PaimonSplitReader keeps the full file list.
  connector::FileDataSource::addSplit(
      PaimonSplitReader::makeFileConnectorSplit(
          *paimonSplit_, paimonSplit_->dataFiles().front()));
}

std::unique_ptr<connector::FileSplitReader>
PaimonDataSource::createSplitReader() {
  if (!paimonSplit_->rawConvertible()) {
    return std::make_unique<PaimonMergeSplitReader>(
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
  paimonSplit_ = std::move(source->paimonSplit_);
  connector::FileDataSource::setFromDataSource(std::move(sourceUnique));
  // Rebind the lazy reader's borrowed stats pointer after ownership transfer.
  // A skipped split has no reader.
  if (splitReader_ != nullptr) {
    checkedPointerCast<PaimonSplitReader>(splitReader_.get())
        ->setRuntimeStats(runtimeStats_);
  }
}

} // namespace facebook::velox::connector::paimon
