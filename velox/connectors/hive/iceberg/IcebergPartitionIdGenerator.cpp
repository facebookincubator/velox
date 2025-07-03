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
#include "velox/connectors/hive/iceberg/Transforms.h"

namespace facebook::velox::connector::hive::iceberg {

IcebergPartitionIdGenerator::IcebergPartitionIdGenerator(
    const RowTypePtr& inputType,
    std::vector<column_index_t> partitionChannels,
    uint32_t maxPartitions,
    memory::MemoryPool* pool,
    const std::shared_ptr<const IcebergInsertTableHandle>& insertTableHandle,
    bool partitionPathAsLowerCase)
    : PartitionIdGenerator(
          partitionChannels,
          maxPartitions,
          partitionPathAsLowerCase),
      pool_(pool),
      insertTableHandle_(insertTableHandle) {
  VELOX_USER_CHECK(pool, "Memory pool cannot be null");
  VELOX_USER_CHECK(insertTableHandle, "insertTableHandle cannot be null");
  std::vector<TypePtr> partitionKeyTypes;
  std::vector<std::string> partitionKeyNames;
  column_index_t i{0};
  for (auto& columnTransform : insertTableHandle->columnTransforms()) {
    hashers_.emplace_back(
        exec::VectorHasher::create(columnTransform.resultType(), i++));
    std::string key = columnTransform.columnName();
    VELOX_USER_CHECK(
        exec::VectorHasher::typeKindSupportsValueIds(
            columnTransform.resultType()->kind()),
        "Unsupported partition type: {}.",
        columnTransform.resultType()->toString());
    partitionKeyTypes.emplace_back(columnTransform.resultType());
    if (columnTransform.transformName() != "identity") {
      key += "_" + columnTransform.transformName();
    }
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
  const int32_t transformCount = insertTableHandle_->columnTransforms().size();
  columns.reserve(transformCount);
  names.reserve(transformCount);
  types.reserve(transformCount);
  for (auto& columnTransform : insertTableHandle_->columnTransforms()) {
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

} // namespace facebook::velox::connector::hive::iceberg
