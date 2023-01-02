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

#include "velox/connectors/hive/PartitionOrdinalIdGenerator.h"

#include "velox/connectors/hive/HivePartitionUtil.h"

namespace facebook::velox::connector::hive {

namespace {

template <TypeKind Kind>
inline void validatePartitionVector(const BaseVector* partitionVector) {
  using T = typename TypeTraits<Kind>::NativeType;
  const auto* simpleVector = partitionVector->as<SimpleVector<T>>();
  VELOX_CHECK_NOT_NULL(
      simpleVector,
      "Partition vector of encoding {} is not supported.",
      VectorEncoding::mapSimpleToName(partitionVector->encoding()));
}

template <TypeKind Kind>
inline uint64_t hashPartitionValue(
    const BaseVector* partitionVector,
    vector_size_t row) {
  using T = typename TypeTraits<Kind>::NativeType;
  const auto* simpleVector = partitionVector->asUnchecked<SimpleVector<T>>();
  return simpleVector->hashValueAt(row);
}

} // namespace

PartitionOrdinalIdGenerator::PartitionOrdinalIdGenerator(
    const RowTypePtr& inputType,
    std::vector<column_index_t> partitionChannels,
    uint32_t maxPartitions,
    memory::MemoryPool* pool)
    : partitionChannels_(std::move(partitionChannels)),
      maxPartitions_(maxPartitions),
      partitionsSet_{
          97,
          PartitionHasher(partitionChannels_),
          PartitionComparer(partitionChannels_)},
      partitionsVector_(BaseVector::create<RowVector>(inputType, 0, pool)) {
  VELOX_CHECK_GT(
      partitionChannels_.size(),
      0,
      "PartitionOrdinalIdGenerator must be applied to a partitioned table.");

  // Assume small value for maxPartitions. Generously initialize
  // partitionsVector_ with capacity maxPartitions to avoid repeated resize.
  for (column_index_t channel : partitionChannels_) {
    partitionsVector_->childAt(channel)->resize(maxPartitions_);
  }
}

void PartitionOrdinalIdGenerator::run(
    const RowVectorPtr& input,
    raw_vector<int32_t>& result) {
  result.resize(input->size());

  for (vector_size_t i = 0; i < partitionChannels_.size(); i++) {
    PARTITION_TYPE_DISPATCH(
        validatePartitionVector,
        input->childAt(partitionChannels_[i])->typeKind(),
        input->childAt(partitionChannels_[i])->loadedVector());
  }

  for (vector_size_t row = 0; row < input->size(); row++) {
    auto entry = partitionsSet_.find({.vector = input, .index = row});
    if (entry != partitionsSet_.end()) {
      result[row] = entry->index;
    } else {
      VELOX_USER_CHECK_LE(
          partitionsSet_.size() + 1,
          maxPartitions_,
          "Exceeded limit of distinct partitions.");

      vector_size_t nextIndex = partitionsSet_.size();

      for (column_index_t channel : partitionChannels_) {
        partitionsVector_->childAt(channel)->copy(
            input->childAt(channel).get(), nextIndex, row, 1);
      }

      partitionsSet_.insert({.vector = partitionsVector_, .index = nextIndex});
      result[row] = nextIndex;
    }
  }
}

PartitionOrdinalIdGenerator::PartitionHasher::PartitionHasher(
    std::vector<column_index_t> partitionChannels)
    : partitionChannels_(std::move(partitionChannels)) {}

size_t PartitionOrdinalIdGenerator::PartitionHasher::operator()(
    const Partition& partition) const {
  uint64_t result = 0;
  for (vector_size_t i = 0; i < partitionChannels_.size(); i++) {
    uint64_t hash = PARTITION_TYPE_DISPATCH(
        hashPartitionValue,
        partition.vector->childAt(partitionChannels_[i])->typeKind(),
        partition.vector->childAt(partitionChannels_[i])->loadedVector(),
        partition.index);
    result = i > 0 ? bits::hashMix(result, hash) : hash;
  }
  return result;
}

PartitionOrdinalIdGenerator::PartitionComparer::PartitionComparer(
    std::vector<column_index_t> partitionChannels)
    : partitionChannels_(std::move(partitionChannels)) {}

bool PartitionOrdinalIdGenerator::PartitionComparer::operator()(
    const Partition& left,
    const Partition& right) const {
  for (column_index_t channel : partitionChannels_) {
    if (!left.vector->childAt(channel)->equalValueAt(
            right.vector->childAt(channel).get(), left.index, right.index)) {
      return false;
    }
  }
  return true;
}

} // namespace facebook::velox::connector::hive
