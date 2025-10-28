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
#include "velox/connectors/hive/iceberg/IcebergPartitionPath.h"
#include "velox/connectors/hive/iceberg/TransformEvaluator.h"
#include "velox/connectors/hive/iceberg/TransformExprBuilder.h"
#include "velox/functions/prestosql/URLFunctions.h"

namespace facebook::velox::connector::hive::iceberg {

namespace {

template <TypeKind Kind>
std::pair<std::string, std::string> makePartitionKeyValueString(
    const IcebergPartitionPathPtr& formatter,
    const BaseVector& partitionVector,
    vector_size_t row,
    const std::string& name,
    const TypePtr& type) {
  using T = typename TypeTraits<Kind>::NativeType;
  if (partitionVector.isNullAt(row)) {
    return std::make_pair(name, "null");
  }

  return std::make_pair(
      name,
      formatter->toPartitionString(
          partitionVector.as<SimpleVector<T>>()->valueAt(row), type));
}

} // namespace

IcebergPartitionIdGenerator::IcebergPartitionIdGenerator(
    const RowTypePtr& inputType,
    std::vector<column_index_t> partitionChannels,
    IcebergPartitionSpecPtr partitionSpec,
    uint32_t maxPartitions,
    const ConnectorQueryCtx* connectorQueryCtx)
    : PartitionIdGenerator(
          partitionChannels,
          maxPartitions,
          connectorQueryCtx->memoryPool(),
          /*partitionPathAsLowerCase=*/false),
      partitionSpec_(partitionSpec),
      connectorQueryCtx_(connectorQueryCtx),
      transformEvaluator_(
          std::make_unique<TransformEvaluator>(
              TransformExprBuilder::toExpressions(
                  partitionSpec_,
                  partitionChannels_,
                  inputType),
              connectorQueryCtx_)) {
  // Build partition key types and names from transforms.
  // For each transform, create a hasher for its result type and build the
  // partition key name (either source column name for identity transforms,
  // or "columnName_transformName" for other transforms).
  std::vector<TypePtr> partitionKeyTypes;
  std::vector<std::string> partitionKeyNames;
  column_index_t idx{0};
  for (const auto& field : partitionSpec_->fields) {
    hashers_.emplace_back(
        exec::VectorHasher::create(field.resultType(), idx++));

    VELOX_USER_CHECK(
        hashers_.back()->typeSupportsValueIds(),
        "Type is not supported as a partition column: {}",
        field.resultType()->toString());

    partitionKeyTypes.emplace_back(field.resultType());
    std::string key = field.transformType == TransformType::kIdentity
        ? field.name
        : fmt::format(
              "{}_{}",
              field.name,
              TransformTypeName::toName(field.transformType));
    partitionKeyNames.emplace_back(std::move(key));
  }

  rowType_ = ROW(std::move(partitionKeyNames), std::move(partitionKeyTypes));

  // At least one partition.
  partitionValues_ = BaseVector::create<RowVector>(
      rowType_, 1, connectorQueryCtx_->memoryPool());
}

void IcebergPartitionIdGenerator::savePartitionValues(
    uint32_t partitionId,
    const RowVectorPtr& input,
    vector_size_t row) {
  if (partitionId >= partitionValues_->size()) {
    auto currentSize = partitionValues_->size();
    auto newSize = std::min<vector_size_t>(currentSize * 2, maxPartitions_);

    partitionValues_->resize(newSize);
    for (auto& child : partitionValues_->children()) {
      child->resize(newSize);
    }
  }

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

  // Step 1: Apply transforms to partition columns.
  auto transformedColumns = transformEvaluator_->evaluate(input);

  // Step 2: Create RowVector based on transformed columns.
  const auto& rowVector = std::make_shared<RowVector>(
      connectorQueryCtx_->memoryPool(),
      rowType_,
      nullptr,
      numRows,
      std::move(transformedColumns));

  // Step 3: Compute value IDs, map to sequential partition IDs and save
  // partition values.
  computeAndSavePartitionIds(rowVector, result);
}

std::vector<std::pair<std::string, std::string>>
IcebergPartitionIdGenerator::extractPartitionKeyValues(
    const RowVectorPtr& partitionsVector,
    vector_size_t row) const {
  std::vector<std::pair<std::string, std::string>> partitionKeyValues;
  VELOX_DCHECK_EQ(
      partitionsVector->childrenSize(),
      partitionSpec_->fields.size(),
      "Partition values and partition transform does not match.");
  for (auto i = 0; i < partitionsVector->childrenSize(); i++) {
    const auto& formatter = std::make_shared<IcebergPartitionPath>(
        partitionSpec_->fields.at(i).transformType);
    const auto& column = partitionsVector->childAt(i);
    partitionKeyValues.push_back(VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
        makePartitionKeyValueString,
        column->typeKind(),
        formatter,
        *column->loadedVector(),
        row,
        asRowType(partitionsVector->type())->nameOf(i),
        column->type()));
  }
  return partitionKeyValues;
}

std::string IcebergPartitionIdGenerator::partitionName(
    uint32_t partitionId) const {
  auto keyValues = extractPartitionKeyValues(partitionValues_, partitionId);
  std::ostringstream out;

  for (auto& [key, value] : keyValues) {
    if (out.tellp() > 0) {
      out << '/';
    }

    if (partitionPathAsLowerCase_) {
      folly::toLowerAscii(key);
    }
    std::string encodedKey;
    std::string encodedValue;

    // Pre-allocate for worst case: every byte is invalid UTF-8.
    // urlEscape() writes directly into the pre-allocated buffer and
    // calls resize() at the end to shrink to the actual size used.
    encodedKey.resize(key.size() * 9);
    encodedValue.resize(value.size() * 9);
    functions::detail::urlEscape(encodedKey, key);
    functions::detail::urlEscape(encodedValue, value);
    out << encodedKey << '=' << encodedValue;
  }

  return out.str();
}

} // namespace facebook::velox::connector::hive::iceberg
