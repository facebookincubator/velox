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

#include "velox/common/base/RawVector.h"
#include "velox/vector/ComplexVector.h"

#include <folly/container/F14Set.h>

namespace facebook::velox::connector::hive {

/// Generate sequential consecutive integer IDs for distinct partition values,
/// which could be used as vector index.
class PartitionOrdinalIdGenerator {
 public:
  // Represent partition by index into a vector.
  struct Partition {
    const RowVectorPtr& vector;
    vector_size_t index;
  };

  class PartitionHasher {
   public:
    PartitionHasher(std::vector<column_index_t> partitionChannels);

    size_t operator()(const Partition& partition) const;

   private:
    const std::vector<column_index_t> partitionChannels_;
  };

  class PartitionComparer {
   public:
    PartitionComparer(std::vector<column_index_t> partitionChannels);

    bool operator()(const Partition& left, const Partition& right) const;

    const std::vector<column_index_t> partitionChannels_;
  };

  /// @param inputType RowType of the input.
  /// @param partitionChannels Channels of partition keys in the input
  /// RowVector.
  /// @param maxPartitions The max number of distinct partitions.
  /// @param pool Memory pool to use for allocation.
  PartitionOrdinalIdGenerator(
      const RowTypePtr& inputType,
      std::vector<column_index_t> partitionChannels,
      uint32_t maxPartitions,
      memory::MemoryPool* pool);

  /// Generate sequential consecutive partition IDs for input vector.
  /// @param input Input RowVector.
  /// @param result Generated integer IDs indexed by input row number.
  void run(const RowVectorPtr& input, raw_vector<int32_t>& result);

 private:
  const std::vector<column_index_t> partitionChannels_;

  const uint32_t maxPartitions_;

  // Keep track of all distinct partitions. Indices are used for the partition
  // IDs to be returned.
  RowVectorPtr partitionsVector_;

  // Used for partition dedup, with elements referring to the partitionsVector_.
  folly::F14FastSet<Partition, PartitionHasher, PartitionComparer>
      partitionsSet_;
};

} // namespace facebook::velox::connector::hive
