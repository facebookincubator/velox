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

#include "velox/connectors/hive/iceberg/IcebergChangelogDataSource.h"

#include <memory>

#include "velox/connectors/hive/iceberg/IcebergSplit.h"

namespace facebook::velox::connector::hive::iceberg {

IcebergChangelogDataSource::IcebergChangelogDataSource(
    std::unique_ptr<connector::DataSource> baseDataSource,
    RowTypePtr changelogOutputType,
    ColumnHandleMap changelogColumnHandles)
    : baseDataSource_(std::move(baseDataSource)),
      changelogOutputType_(std::move(changelogOutputType)),
      changelogColumnHandles_(
          std::make_shared<ColumnHandleMap>(
              std::move(changelogColumnHandles))) {}

void IcebergChangelogDataSource::addSplit(
    std::shared_ptr<connector::ConnectorSplit> split) {
  auto icebergSplit = std::dynamic_pointer_cast<HiveIcebergSplit>(split);
  VELOX_CHECK_NOT_NULL(icebergSplit, "icebergSplit is null");
  // Store the changelogSplitInfo so we can access during transformation
  changelogSplitInfo_ = icebergSplit->changelogSplitInfo;

  // Delegate to the wrapped data source
  baseDataSource_->addSplit(std::move(split));
}

std::optional<RowVectorPtr> IcebergChangelogDataSource::next(
    uint64_t size,
    velox::ContinueFuture& future) {
  // Get data from the wrapped data source
  auto dataVector = baseDataSource_->next(size, future);

  if (!dataVector.has_value()) {
    return std::nullopt;
  }

  if (!dataVector.value()) {
    return nullptr;
  }

  // Transform the data output to changelog format
  auto* rowVector = dataVector.value()->as<RowVector>();
  VELOX_CHECK_NOT_NULL(
      rowVector, "Expected RowVector output from base data source");

  auto transformedVector =
      transformChangelogOutput(dataVector.value(), rowVector->pool());

  return std::dynamic_pointer_cast<RowVector>(transformedVector);
}

VectorPtr IcebergChangelogDataSource::transformChangelogOutput(
    const VectorPtr& dataOutput,
    memory::MemoryPool* pool) {
  if (!changelogSplitInfo_) {
    // Not a changelog split, return data as-is
    return dataOutput;
  }

  auto* dataRowVector = dataOutput->as<RowVector>();
  VELOX_CHECK_NOT_NULL(dataRowVector, "Expected RowVector output");

  const auto positionCount = dataRowVector->size();
  std::vector<VectorPtr> changelogColumns;
  changelogColumns.reserve(changelogOutputType_->size());

  // Build changelog columns in the order specified by changelogOutputType_
  for (size_t i = 0; i < changelogOutputType_->size(); ++i) {
    const auto& typeName = changelogOutputType_->nameOf(i);
    auto it = changelogColumnHandles_->find(typeName);
    VELOX_CHECK(
        it != changelogColumnHandles_->end(),
        "Column handle not found for changelog field: {}",
        typeName);
    const auto& fieldName = it->second->name();

    if (fieldName == "operation") {
      std::string operationStr;
      switch (changelogSplitInfo_->operation) {
        case ChangelogOperation::INSERT:
          operationStr = "INSERT";
          break;
        case ChangelogOperation::DELETE:
          operationStr = "DELETE";
          break;
        case ChangelogOperation::UPDATE_BEFORE:
          operationStr = "UPDATE_BEFORE";
          break;
        case ChangelogOperation::UPDATE_AFTER:
          operationStr = "UPDATE_AFTER";
          break;
        default:
          VELOX_FAIL("Unknown changelog operation");
      }
      auto constant = BaseVector::createConstant(
          VARCHAR(), variant(operationStr), positionCount, pool);
      changelogColumns.push_back(constant);
    } else if (fieldName == "ordinal") {
      auto constant = BaseVector::createConstant(
          BIGINT(), variant(changelogSplitInfo_->ordinal), positionCount, pool);
      changelogColumns.push_back(constant);
    } else if (fieldName == "snapshotid") {
      auto constant = BaseVector::createConstant(
          BIGINT(),
          variant(changelogSplitInfo_->snapshotId),
          positionCount,
          pool);
      changelogColumns.push_back(constant);
    } else if (fieldName == "rowdata") {
      // Wrap all data columns into a RowVector
      // Use the rowdata type from changelogOutputType_, not
      // dataRowVector->type()
      auto rowdataType = changelogOutputType_->childAt(i);
      auto rowdataRowType =
          std::dynamic_pointer_cast<const RowType>(rowdataType);
      VELOX_CHECK_NOT_NULL(rowdataRowType, "rowdata type must be a RowType");

      // Validate that the rowdata type matches the data output structure
      VELOX_CHECK_EQ(
          rowdataRowType->size(),
          dataRowVector->childrenSize(),
          "rowdata type child count ({}) does not match data output child count ({})",
          rowdataRowType->size(),
          dataRowVector->childrenSize());

      for (size_t j = 0; j < rowdataRowType->size(); ++j) {
        VELOX_CHECK(
            rowdataRowType->childAt(j)->equivalent(
                *dataRowVector->childAt(j)->type()),
            "rowdata type child {} ({}) does not match data output child {} ({})",
            j,
            rowdataRowType->childAt(j)->toString(),
            j,
            dataRowVector->childAt(j)->type()->toString());
      }

      auto rowdataVector = std::make_shared<RowVector>(
          pool,
          rowdataType,
          BufferPtr(nullptr),
          positionCount,
          dataRowVector->children());
      changelogColumns.push_back(rowdataVector);
    }
  }

  return std::make_shared<RowVector>(
      pool,
      changelogOutputType_,
      BufferPtr(nullptr),
      positionCount,
      std::move(changelogColumns));
}

void IcebergChangelogDataSource::addDynamicFilter(
    column_index_t outputChannel,
    const std::shared_ptr<common::Filter>& filter) {
  // Delegate to the wrapped data source
  baseDataSource_->addDynamicFilter(outputChannel, filter);
}

uint64_t IcebergChangelogDataSource::getCompletedBytes() {
  return baseDataSource_->getCompletedBytes();
}

uint64_t IcebergChangelogDataSource::getCompletedRows() {
  return baseDataSource_->getCompletedRows();
}

std::unordered_map<std::string, RuntimeMetric>
IcebergChangelogDataSource::getRuntimeStats() {
  return baseDataSource_->getRuntimeStats();
}

bool IcebergChangelogDataSource::allPrefetchIssued() const {
  return baseDataSource_->allPrefetchIssued();
}

void IcebergChangelogDataSource::setFromDataSource(
    std::unique_ptr<connector::DataSource> sourceUnique) {
  auto changelogSource =
      dynamic_cast<IcebergChangelogDataSource*>(sourceUnique.get());
  VELOX_CHECK_NOT_NULL(changelogSource, "changelogSource is null");
  // Transfer the wrapped data source
  baseDataSource_->setFromDataSource(
      std::move(changelogSource->baseDataSource_));
  changelogOutputType_ = std::move(changelogSource->changelogOutputType_);
  changelogColumnHandles_ = std::move(changelogSource->changelogColumnHandles_);
  changelogSplitInfo_ = std::move(changelogSource->changelogSplitInfo_);
}

} // namespace facebook::velox::connector::hive::iceberg
