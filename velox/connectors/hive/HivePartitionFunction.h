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
#pragma once

#include "velox/core/PlanNode.h"
#include "velox/vector/DecodedVector.h"

namespace facebook::velox::connector::hive {

class HivePartitionFunction : public core::PartitionFunction {
 public:
  HivePartitionFunction(
      int numBuckets,
      std::vector<int> bucketToPartition,
      std::vector<ChannelIndex> keyChannels);

  ~HivePartitionFunction() override = default;

  void partition(const RowVector& input, std::vector<uint32_t>& partitions)
      override;

  static void hash(
      const DecodedVector& values,
      TypeKind typeKind,
      vector_size_t size,
      bool mix,
      std::vector<uint32_t>& hashes);

  inline static uint32_t mix(uint32_t hash1, uint32_t hash2) {
    return hash1 * 31 + hash2;
  }

 private:
  const int numBuckets_;
  const std::vector<int> bucketToPartition_;
  const std::vector<ChannelIndex> keyChannels_;

  // Reusable memory.
  std::vector<uint32_t> hashes_;
  SelectivityVector rows_;
  std::vector<DecodedVector> decodedVectors_;
};
} // namespace facebook::velox::connector::hive
