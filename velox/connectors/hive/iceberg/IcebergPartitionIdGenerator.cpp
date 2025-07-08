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

// Iceberg spec requires URL encoding in the partition path.
std::string UrlEncode(const std::string& data) {
  std::string ret;
  ret.reserve(data.size() * 3);

  for (unsigned char c : data) {
    if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '*') {
      // These characters are not encoded in Java's URLEncoder.
      ret += c;
    } else if (c == ' ') {
      // Space is converted to '+' in Java's URLEncoder.
      ret += '+';
    } else {
      // All other characters are percent-encoded.
      ret += fmt::format("%{:02X}", c);
    }
  }

  return ret;
}

std::string toLower(const std::string& data) {
  std::string ret;
  ret.reserve(data.size());
  std::transform(
      data.begin(), data.end(), std::back_inserter(ret), [](auto& c) {
        return std::tolower(c);
      });
  return ret;
}

} // namespace

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
  auto pairs = extractPartitionKeyValues(partitionValues_, partitionId);

  size_t estimatedSize = 0;
  for (const auto& pair : pairs) {
    estimatedSize += pair.first.size() + pair.second.size() + 2;
  }

  std::string ret;
  ret.reserve(estimatedSize * 1.5);

  for (const auto& pair : pairs) {
    if (!ret.empty()) {
      ret.push_back('/');
    }
    ret += partitionPathAsLowerCase_ ? UrlEncode(toLower(pair.first))
                                     : UrlEncode(pair.first);
    ret.push_back('=');
    if (pair.second.empty()) {
      ret += nullValueName;
    } else {
      TypePtr type = nullptr;
      for (const auto& columnTransform : columnTransforms_) {
        if (columnTransform.columnName() == pair.first ||
            (columnTransform.transformName() == "trunc" &&
             fmt::format("{}_{}", columnTransform.columnName(), "trunc") ==
                 pair.first)) {
          type = columnTransform.resultType();
          break;
        }
      }

      if (type && type->isDecimal()) {
        ret += DecimalUtil::toString(HugeInt::parse(pair.second), type);
      } else {
        ret += UrlEncode(pair.second);
      }
    }
  }

  return ret;
}

} // namespace facebook::velox::connector::hive::iceberg
