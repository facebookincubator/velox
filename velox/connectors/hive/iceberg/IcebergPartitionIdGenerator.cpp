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

#include "velox/connectors/hive/HivePartitionUtil.h"
#include "velox/connectors/hive/iceberg/Transforms.h"

namespace facebook::velox::connector::hive::iceberg {

namespace {

template <TypeKind Kind>
std::pair<std::string, std::string> makePartitionKeyValueString(
    const BaseVector* partitionVector,
    vector_size_t row,
    const std::string& name,
    const std::shared_ptr<Transform>& columnTransform) {
  using T = typename TypeTraits<Kind>::NativeType;
  if (partitionVector->as<SimpleVector<T>>()->isNullAt(row)) {
    return std::make_pair(name, "null");
  }

  return std::make_pair(
      name,
      columnTransform->toHumanString(
          partitionVector->as<SimpleVector<T>>()->valueAt(row)));
}

// Iceberg spec requires URL encoding in the partition path.
// This function matches java.net.URLEncoder.encode(string, "UTF-8").
std::string urlEncode(const StringView& data) {
  std::ostringstream ret;

  for (unsigned char c : data) {
    // These characters are not encoded in Java's URLEncoder.
    if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '*') {
      ret << c;
    } else if (c == ' ') {
      ret << '+';
    } else {
      // All other characters are percent-encoded.
      ret << fmt::format("%{:02X}", c);
    }
  }

  return ret.str();
}

} // namespace

IcebergPartitionIdGenerator::IcebergPartitionIdGenerator(
    std::vector<column_index_t> partitionChannels,
    uint32_t maxPartitions,
    memory::MemoryPool* pool,
    const std::vector<std::shared_ptr<Transform>>& columnTransforms,
    bool partitionPathAsLowerCase)
    : PartitionIdGenerator(
          partitionChannels,
          maxPartitions,
          pool,
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
        exec::VectorHasher::create(columnTransform->resultType(), i++));
    VELOX_USER_CHECK(
        exec::VectorHasher::typeKindSupportsValueIds(
            columnTransform->resultType()->kind()),
        "Unsupported partition type: {}.",
        columnTransform->resultType()->toString());
    partitionKeyTypes.emplace_back(columnTransform->resultType());
    std::string key =
        columnTransform->transformType() == TransformType::kIdentity
        ? columnTransform->sourceColumnName()
        : fmt::format(
              "{}_{}",
              columnTransform->sourceColumnName(),
              columnTransform->name());
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
    uint32_t partitionId,
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
    names.emplace_back(columnTransform->sourceColumnName());
    types.emplace_back(columnTransform->resultType());
    columns.emplace_back(columnTransform->transform(input));
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

std::vector<std::pair<std::string, std::string>>
IcebergPartitionIdGenerator::extractPartitionKeyValues(
    const RowVectorPtr& partitionsVector,
    vector_size_t row) const {
  std::vector<std::pair<std::string, std::string>> partitionKeyValues;
  VELOX_DCHECK_EQ(
      partitionsVector->childrenSize(),
      columnTransforms_.size(),
      "Partition values and partition transform doe not match.");
  for (auto i = 0; i < partitionsVector->childrenSize(); i++) {
    partitionKeyValues.push_back(VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
        makePartitionKeyValueString,
        partitionsVector->childAt(i)->typeKind(),
        partitionsVector->childAt(i)->loadedVector(),
        row,
        asRowType(partitionsVector->type())->nameOf(i),
        columnTransforms_[i]));
  }
  return partitionKeyValues;
}

std::string IcebergPartitionIdGenerator::partitionName(
    uint64_t partitionId) const {
  auto keyValues = extractPartitionKeyValues(partitionValues_, partitionId);
  std::ostringstream ret;

  for (auto& [key, value] : keyValues) {
    if (ret.tellp() > 0) {
      ret << '/';
    }

    if (partitionPathAsLowerCase_) {
      folly::toLowerAscii(key);
    }
    ret << urlEncode(key.data()) << '=' << urlEncode(value.data());
  }

  return ret.str();
}

} // namespace facebook::velox::connector::hive::iceberg
