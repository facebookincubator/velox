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

#include <optional>

#include "velox/connectors/clp/ClpColumnHandle.h"
#include "velox/connectors/clp/ClpConnectorSplit.h"
#include "velox/connectors/clp/ClpDataSource.h"
#include "velox/connectors/clp/ClpTableHandle.h"
#include "velox/connectors/clp/search_lib/ClpCursor.h"
#include "velox/connectors/clp/search_lib/ClpVectorLoader.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::connector::clp {

ClpDataSource::ClpDataSource(
const RowTypePtr& outputType,
  const ConnectorTableHandlePtr& tableHandle,
  const connector::ColumnHandleMap& columnHandles,
  velox::memory::MemoryPool* pool,
  std::shared_ptr<const ClpConfig>& clpConfig)
    : pool_(pool), outputType_(outputType) {
  auto clpTableHandle = std::dynamic_pointer_cast<const ClpTableHandle>(tableHandle);
  storageType_ = clpConfig->storageType();

  for (const auto& outputName : outputType->names()) {
    auto columnHandle = columnHandles.find(outputName);
    VELOX_CHECK(
        columnHandle != columnHandles.end(),
        "ColumnHandle not found for output name: {}",
        outputName);
    auto clpColumnHandle =
        std::dynamic_pointer_cast<const ClpColumnHandle>(columnHandle->second);
    VELOX_CHECK_NOT_NULL(
        clpColumnHandle,
        "ColumnHandle must be an instance of ClpColumnHandle for output name: {}",
        outputName);
    auto columnName = clpColumnHandle->originalColumnName();
    auto columnType = clpColumnHandle->columnType();
    addFieldsRecursively(columnType, columnName);
  }
}

void ClpDataSource::addFieldsRecursively(
    const TypePtr& columnType,
    const std::string& parentName) {
  if (columnType->kind() == TypeKind::ROW) {
    const auto& rowType = columnType->asRow();
    for (uint32_t i = 0; i < rowType.size(); ++i) {
      const auto& childType = rowType.childAt(i);
      const auto childName = parentName + "." + rowType.nameOf(i);
      addFieldsRecursively(childType, childName);
    }
  } else {
    search_lib::ColumnType clpColumnType = search_lib::ColumnType::Unknown;
    switch (columnType->kind()) {
      case TypeKind::BOOLEAN:
        clpColumnType = search_lib::ColumnType::Boolean;
        break;
      case TypeKind::INTEGER:
      case TypeKind::BIGINT:
      case TypeKind::SMALLINT:
      case TypeKind::TINYINT:
        clpColumnType = search_lib::ColumnType::Integer;
        break;
      case TypeKind::DOUBLE:
      case TypeKind::REAL:
        clpColumnType = search_lib::ColumnType::Float;
        break;
      case TypeKind::VARCHAR:
        clpColumnType = search_lib::ColumnType::String;
        break;
      case TypeKind::ARRAY:
        clpColumnType = search_lib::ColumnType::Array;
        break;
      case TypeKind::TIMESTAMP:
        clpColumnType = search_lib::ColumnType::Timestamp;
        break;
      default:
        VELOX_USER_FAIL("Type not supported: {}", columnType->name());
    }
    fields_.emplace_back(search_lib::Field{clpColumnType, parentName});
  }
}

void ClpDataSource::addSplit(std::shared_ptr<ConnectorSplit> split) {
  auto clpSplit = std::dynamic_pointer_cast<ClpConnectorSplit>(split);

  if (storageType_ == ClpConfig::StorageType::kFs) {
    cursor_ = std::make_unique<search_lib::ClpCursor>(
        clp_s::InputSource::Filesystem, clpSplit->path_);
  } else if (storageType_ == ClpConfig::StorageType::kS3) {
    cursor_ = std::make_unique<search_lib::ClpCursor>(
        clp_s::InputSource::Network, clpSplit->path_);
  }

  auto pushDownQuery = clpSplit->kqlQuery_;
  if (pushDownQuery && !pushDownQuery->empty()) {
    cursor_->executeQuery(*pushDownQuery, fields_);
  } else {
    cursor_->executeQuery("*", fields_);
  }
}

VectorPtr ClpDataSource::createVector(
    const TypePtr& vectorType,
    size_t vectorSize,
    const std::vector<clp_s::BaseColumnReader*>& projectedColumns,
    const std::shared_ptr<std::vector<uint64_t>>& filteredRows,
    size_t& readerIndex) {
  if (vectorType->kind() == TypeKind::ROW) {
    std::vector<VectorPtr> children;
    auto& rowType = vectorType->as<TypeKind::ROW>();
    for (uint32_t i = 0; i < rowType.size(); ++i) {
      children.push_back(createVector(
          rowType.childAt(i),
          vectorSize,
          projectedColumns,
          filteredRows,
          readerIndex));
    }
    return std::make_shared<RowVector>(
        pool_, vectorType, nullptr, vectorSize, std::move(children));
  }
  auto vector = BaseVector::create(vectorType, vectorSize, pool_);
  vector->setNulls(allocateNulls(vectorSize, pool_, bits::kNull));

  VELOX_CHECK_LT(
      readerIndex, projectedColumns.size(), "Reader index out of bounds");
  auto projectedColumn = projectedColumns[readerIndex];
  auto projectedType = fields_[readerIndex].type;
  readerIndex++;
  return std::make_shared<LazyVector>(
      pool_,
      vectorType,
      vectorSize,
      std::make_unique<search_lib::ClpVectorLoader>(
          projectedColumn, projectedType, filteredRows),
      std::move(vector));
}

std::optional<RowVectorPtr> ClpDataSource::next(
    uint64_t size,
    ContinueFuture& future) {
  auto filteredRows = std::make_shared<std::vector<uint64_t>>();
  auto rowsScanned = cursor_->fetchNext(size, filteredRows);
  auto rowsFiltered = filteredRows->size();
  if (rowsFiltered == 0) {
    return nullptr;
  }
  completedRows_ += rowsScanned;
  size_t readerIndex = 0;
  const auto& projectedColumns = cursor_->getProjectedColumns();
  VELOX_CHECK_EQ(
      projectedColumns.size(),
      fields_.size(),
      "Projected columns size {} does not match fields size {}",
      projectedColumns.size(),
      fields_.size());
  return std::dynamic_pointer_cast<RowVector>(createVector(
      outputType_, rowsFiltered, projectedColumns, filteredRows, readerIndex));
}

} // namespace facebook::velox::connector::clp
