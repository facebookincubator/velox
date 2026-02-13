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

#include "velox/connectors/hive/PartitionIdGenerator.h"
#include "velox/connectors/hive/iceberg/IcebergDataSink.h"

namespace facebook::velox::connector::hive::iceberg {
class IcebergPartitionIdGenerator : public PartitionIdGenerator {
 public:
  IcebergPartitionIdGenerator(
      std::vector<column_index_t> partitionChannels,
      uint32_t maxPartitions,
      memory::MemoryPool* pool,
      const std::vector<std::shared_ptr<Transform>>& columnTransforms,
      bool partitionPathAsLowerCase);

  /// Generate sequential partition IDs for input vector.
  /// @param input Input RowVector.
  /// @param result Generated integer IDs indexed by input row number.
  void run(const RowVectorPtr& input, raw_vector<uint64_t>& result) override;

  /// Return partition name for the given partition id in the typical Hive
  /// style. It is derived from the partitionValues_ at index partitionId.
  /// Partition keys appear in the order of partition columns in the table
  /// schema.
  std::string partitionName(uint64_t partitionId) const override;

  /// Return the partition values for all partitions.
  RowVectorPtr partitionValues() const {
    return partitionValues_;
  }

 private:
  void savePartitionValues(
      uint32_t partitionId,
      const RowVectorPtr& input,
      vector_size_t row) override;

  std::vector<std::pair<std::string, std::string>> extractPartitionKeyValues(
      const RowVectorPtr& partitionsVector,
      vector_size_t row) const;

  memory::MemoryPool* pool_;
  const std::vector<std::shared_ptr<Transform>> columnTransforms_;
};

} // namespace facebook::velox::connector::hive::iceberg
