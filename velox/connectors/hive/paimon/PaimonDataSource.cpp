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

#include <algorithm>

#include "velox/common/Casts.h"
#include "velox/connectors/hive/paimon/PaimonColumnHandle.h"
#include "velox/connectors/hive/paimon/PaimonConnectorSplit.h"
#include "velox/connectors/hive/paimon/PaimonSplitReader.h"
#include "velox/connectors/hive/paimon/PaimonSubfield.h"
#include "velox/connectors/hive/paimon/PaimonTableHandle.h"
#include "velox/expression/FieldReference.h"

namespace facebook::velox::connector::hive::paimon {
namespace {

// Bind explicit query columns before FileDataSource consumes their names,
// types and roles. Subsequent splits reuse this fixed binding.
const ColumnHandleMap& bindColumns(
    const PaimonTableHandle& table,
    const ColumnHandleMap& assignments) {
  const auto bind = [&](const ColumnHandlePtr& column) {
    VELOX_USER_CHECK_NOT_NULL(column, "Missing Paimon column handle");
    const auto* handle =
        checkedPointerCast<const PaimonColumnHandle>(column.get());
    const auto index = table.columnIndex(handle->fieldId());
    const auto& name = table.rowType()->nameOf(index);
    const auto& type = table.rowType()->childAt(index);
    VELOX_USER_CHECK_EQ(
        handle->name(),
        name,
        "Paimon field ID {} is column '{}' in table '{}' schema {}",
        handle->fieldId(),
        name,
        table.name(),
        table.schemaId());
    VELOX_USER_CHECK(
        *handle->schemaType() == *type,
        "Paimon column '{}' (field ID {}) has type {}, but table '{}' schema {} requires {}",
        name,
        handle->fieldId(),
        handle->schemaType()->toString(),
        table.name(),
        table.schemaId(),
        type->toString());
    const bool partition =
        std::find(
            table.partitionKeys().begin(), table.partitionKeys().end(), name) !=
        table.partitionKeys().end();
    const auto role = partition ? FileColumnHandle::ColumnType::kPartitionKey
                                : FileColumnHandle::ColumnType::kRegular;
    VELOX_USER_CHECK_EQ(
        handle->columnType(),
        role,
        "Paimon column '{}' (field ID {}) has the wrong role in table '{}' schema {}",
        name,
        handle->fieldId(),
        table.name(),
        table.schemaId());
  };
  for (const auto& [_, handle] : assignments) {
    bind(handle);
  }
  for (const auto& handle : table.filterColumnHandles()) {
    bind(handle);
  }
  return assignments;
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
          tableHandle,
          bindColumns(*tableHandle, assignments),
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

  // Include filters extracted from remainingFilter by FileDataSource.
  for (const auto& [path, _] : fileScanState_->filters) {
    resolveSubfieldType(tableHandle->rowType(), path);
  }
  if (const auto* expressions = remainingFilterExprSet()) {
    // The compiled expression already resolves lambda captures and shadowing.
    for (const auto* field : expressions->expr(0)->distinctFields()) {
      const auto index =
          tableHandle->rowType()->getChildIdxIfExists(field->field());
      VELOX_USER_CHECK(
          index.has_value(),
          "Unknown Paimon filter column '{}' in table '{}' schema {}",
          field->field(),
          tableHandle->name(),
          tableHandle->schemaId());
      VELOX_USER_CHECK(
          *field->type() == *tableHandle->rowType()->childAt(*index),
          "Paimon filter column '{}' (field ID {}) has type {}, but schema {} requires {}",
          field->field(),
          tableHandle->fieldIds()[*index],
          field->type()->toString(),
          tableHandle->schemaId(),
          tableHandle->rowType()->childAt(*index)->toString());
    }
  }
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
