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

#include "velox/connectors/hive/iceberg/IcebergPartitionIdGenerator.h"

#include <connectors/hive/HivePartitionUtil.h>
#include <dwio/catalog/fbhive/FileUtils.h>

#include "velox/connectors/hive/iceberg/Transforms.h"

namespace facebook::velox::connector::hive::iceberg {

IcebergPartitionIdGenerator::IcebergPartitionIdGenerator(
    std::vector<column_index_t> partitionChannels,
    uint32_t maxPartitions,
    memory::MemoryPool* pool,
    const std::vector<ColumnTransform>& columnTransforms,
    bool partitionPathAsLowerCase)
    : PartitionIdGenerator(
          partitionChannels,
          maxPartitions,
          partitionPathAsLowerCase),
      pool_(pool),
      columnTransforms_(columnTransforms) {
  VELOX_USER_CHECK_GT(
      columnTransforms_.size(), 0, "columnTransforms_ cannot be null");
  std::vector<TypePtr> partitionKeyTypes;
  std::vector<std::string> partitionKeyNames;
  column_index_t i{0};
  for (const auto& columnTransform : columnTransforms_) {
    hashers_.emplace_back(
        exec::VectorHasher::create(columnTransform.resultType(), i++));
    VELOX_USER_CHECK(
        exec::VectorHasher::typeKindSupportsValueIds(
            columnTransform.resultType()->kind()),
        "Unsupported partition type: {}.",
        columnTransform.resultType()->toString());
    partitionKeyTypes.emplace_back(columnTransform.resultType());
    std::string key = columnTransform.transformName() == "identity"
        ? columnTransform.columnName()
        : fmt::format(
              "{}_{}",
              columnTransform.columnName(),
              columnTransform.transformName());
    partitionKeyNames.emplace_back(std::move(key));
  }
  partitionValues_ = BaseVector::create<RowVector>(
      ROW(std::move(partitionKeyNames), std::move(partitionKeyTypes)),
      maxPartitions,
      pool_);
  for (auto& key : partitionValues_->children()) {
    key->resize(maxPartitions);
  }
}

void IcebergPartitionIdGenerator::savePartitionValues(
    uint64_t partitionId,
    const RowVectorPtr& input,
    vector_size_t row) {
  for (auto i = 0; i < partitionChannels_.size(); ++i) {
    partitionValues_->childAt(i)->copy(
        input->childAt(i).get(), partitionId, row, 1);
  }
}

void IcebergPartitionIdGenerator::run(
    const RowVectorPtr& input,
    raw_vector<uint64_t>& result) {
  const auto numRows = input->size();
  result.resize(numRows);
  std::vector<VectorPtr> columns;
  std::vector<std::string> names;
  std::vector<TypePtr> types;
  const int32_t transformCount = columnTransforms_.size();
  columns.reserve(transformCount);
  names.reserve(transformCount);
  types.reserve(transformCount);
  for (const auto& columnTransform : columnTransforms_) {
    names.emplace_back(columnTransform.columnName());
    types.emplace_back(columnTransform.resultType());
    columns.emplace_back(columnTransform.transform(input));
  }
  const auto rowVector = std::make_shared<RowVector>(
      pool_,
      ROW(std::move(names), std::move(types)),
      nullptr,
      numRows,
      columns);

  // Compute value IDs using VectorHashers and store these in 'result'.
  computeValueIds(rowVector, result);

  // Convert value IDs in 'result' into partition IDs using partitionIds
  // mapping. Update 'result' in place.
  for (auto i = 0; i < numRows; ++i) {
    auto valueId = result[i];
    if (auto it = partitionIds_.find(valueId); it != partitionIds_.end()) {
      result[i] = it->second;
    } else {
      uint64_t nextPartitionId = partitionIds_.size();
      VELOX_USER_CHECK_LT(
          nextPartitionId,
          maxPartitions_,
          "Exceeded limit of {} distinct partitions.",
          maxPartitions_);

      partitionIds_.emplace(valueId, nextPartitionId);
      savePartitionValues(nextPartitionId, rowVector, i);
      result[i] = nextPartitionId;
    }
  }
}

std::string IcebergPartitionIdGenerator::partitionName(
    uint64_t partitionId,
    const std::string& nullValueName) const {
  return dwio::catalog::fbhive::FileUtils::makePartName(
      extractPartitionKeyValues(partitionValues_, partitionId, nullValueName),
      partitionPathAsLowerCase_);
}

} // namespace facebook::velox::connector::hive::iceberg
