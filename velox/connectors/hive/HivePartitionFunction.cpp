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
#include "velox/connectors/hive/HivePartitionFunction.h"

#include <utility>

#include "velox/functions/lib/HiveHash.h"

namespace facebook::velox::connector::hive {

namespace {
using facebook::velox::functions::HiveHash;

void hashPrecomputed(
    uint32_t precomputedHash,
    vector_size_t numRows,
    bool mix,
    std::vector<uint32_t>& hashes) {
  for (auto i = 0; i < numRows; ++i) {
    HiveHash::mergeHash(mix, precomputedHash, hashes[i]);
  }
}
} // namespace

HivePartitionFunction::HivePartitionFunction(
    int numBuckets,
    std::vector<int> bucketToPartition,
    std::vector<column_index_t> keyChannels,
    const std::vector<VectorPtr>& constValues)
    : numBuckets_{numBuckets},
      bucketToPartition_{std::move(bucketToPartition)},
      keyChannels_{std::move(keyChannels)} {
  precomputedHashes_.resize(keyChannels_.size());
  size_t constChannel{0};
  for (auto i = 0; i < keyChannels_.size(); ++i) {
    if (keyChannels_[i] == kConstantChannel) {
      precompute(*(constValues[constChannel++]), i);
    }
  }
}

std::optional<uint32_t> HivePartitionFunction::partition(
    const RowVector& input,
    std::vector<uint32_t>& partitions) {
  const auto numRows = input.size();

  auto& decodedVector = getDecodedVector();
  auto& rows = getRows();
  auto& hashes = getHashes();
  rows.resizeFill(numRows, true);
  if (numRows > hashes.size()) {
    hashes.resize(numRows);
  }
  partitions.resize(numRows);
  HiveHash hiveHash(hashesPool_, rowsPool_, decodedVectorsPool_);
  for (auto i = 0; i < keyChannels_.size(); ++i) {
    if (keyChannels_[i] != kConstantChannel) {
      const auto& keyVector = input.childAt(keyChannels_[i]);
      decodedVector.decode(*keyVector, rows);
      hiveHash.hash(decodedVector, rows, i > 0, hashes, 1);
    } else {
      hashPrecomputed(precomputedHashes_[i], numRows, i > 0, hashes);
    }
  }

  static constexpr int32_t kInt32Max = std::numeric_limits<int32_t>::max();

  if (bucketToPartition_.empty()) {
    // NOTE: if bucket to partition mapping is empty, then we do
    // identical mapping.
    for (auto i = 0; i < numRows; ++i) {
      partitions[i] = (hashes[i] & kInt32Max) % numBuckets_;
    }
  } else {
    for (auto i = 0; i < numRows; ++i) {
      partitions[i] =
          bucketToPartition_[((hashes[i] & kInt32Max) % numBuckets_)];
    }
  }

  return std::nullopt;
}

void HivePartitionFunction::precompute(
    const BaseVector& value,
    size_t channelIndex) {
  if (value.isNullAt(0)) {
    precomputedHashes_[channelIndex] = 0;
    return;
  }

  const SelectivityVector rows(1, true);
  DecodedVector& decodedVector = getDecodedVector();
  decodedVector.decode(value, rows);

  std::vector<uint32_t> hashes{1};
  HiveHash hiveHash(hashesPool_, rowsPool_, decodedVectorsPool_);
  hiveHash.hash(decodedVector, rows, false, hashes, 1);
  precomputedHashes_[channelIndex] = hashes[0];
}

DecodedVector& HivePartitionFunction::getDecodedVector(size_t poolIndex) {
  while (poolIndex >= decodedVectorsPool_.size()) {
    decodedVectorsPool_.push_back(std::make_unique<DecodedVector>());
  }

  return *decodedVectorsPool_[poolIndex];
}

SelectivityVector& HivePartitionFunction::getRows(size_t poolIndex) {
  while (poolIndex >= rowsPool_.size()) {
    rowsPool_.push_back(std::make_unique<SelectivityVector>());
  }

  return *rowsPool_[poolIndex];
}

std::vector<uint32_t>& HivePartitionFunction::getHashes(size_t poolIndex) {
  while (poolIndex >= hashesPool_.size()) {
    hashesPool_.push_back(std::make_unique<std::vector<uint32_t>>());
  }

  return *hashesPool_[poolIndex];
}

HiveIdentityPartitionFunction::HiveIdentityPartitionFunction(
    int numBuckets,
    column_index_t keyChannel)
    : numBuckets_(numBuckets), keyChannel_(keyChannel) {}

std::optional<uint32_t> HiveIdentityPartitionFunction::partition(
    const RowVector& input,
    std::vector<uint32_t>& partitions) {
  const auto& keyVector = input.childAt(keyChannel_);
  VELOX_CHECK(
      keyVector->typeKind() == TypeKind::BIGINT,
      "Only BIGINT type is supported for identity partition function");
  decodedVector_->decode(*keyVector);

  const auto numRows = input.size();
  partitions.resize(numRows);
  for (auto i = 0; i < numRows; i++) {
    partitions[i] = decodedVector_->valueAt<int64_t>(i) % numBuckets_;
  }
  return std::nullopt;
}

} // namespace facebook::velox::connector::hive
