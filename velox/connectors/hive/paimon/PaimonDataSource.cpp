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
  // TODO: Implement primary-key read semantics.
  VELOX_USER_CHECK_EQ(
      tableHandle->tableType(),
      PaimonTableType::kAppendOnly,
      "Paimon same-schema append reader does not support primary-key tables; "
      "primary-key reading is not implemented yet");
}

void PaimonDataSource::addSplit(std::shared_ptr<ConnectorSplit> split) {
  auto paimonSplit = checkedPointerCast<PaimonConnectorSplit>(split);
  VELOX_USER_CHECK_EQ(
      paimonSplit->tableType(),
      paimonTableHandle_->tableType(),
      "Paimon split table type does not match table handle");
  VELOX_USER_CHECK(
      paimonSplit->rawConvertible(),
      "Paimon same-schema append reader requires rawConvertible=true; "
      "merge-on-read and data-evolution column merging are not implemented yet");
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
  paimonSplit_ = std::move(paimonSplit);

  // Use the first file to enter the FileDataSource scan pipeline;
  // PaimonSplitReader keeps the full file list.
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
  paimonSplit_ = std::move(source->paimonSplit_);
  FileDataSource::setFromDataSource(std::move(sourceUnique));
  // Rebind the lazy reader's borrowed stats pointer after ownership transfer.
  checkedPointerCast<PaimonSplitReader>(splitReader_.get())
      ->setRuntimeStats(runtimeStats_);
}

} // namespace facebook::velox::connector::hive::paimon
