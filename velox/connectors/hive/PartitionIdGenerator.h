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

#include "velox/exec/VectorHasher.h"

namespace facebook::velox::connector::hive {
/// Generate sequential integer IDs for distinct partition values, which could
/// be used as vector index.
class PartitionIdGenerator {
 public:
  /// @param inputType RowType of the input.
  /// @param partitionChannels Channels of partition keys in the input
  /// RowVector.
  /// @param maxPartitions The max number of distinct partitions.
  /// @param pool Memory pool. Used to allocate memory for storing unique
  /// partition key values.
  /// @param partitionPathAsLowerCase Used to control whether the partition path
  /// need to convert to lower case.
  PartitionIdGenerator(
      const RowTypePtr& inputType,
      std::vector<column_index_t> partitionChannels,
      uint32_t maxPartitions,
      memory::MemoryPool* pool,
      bool partitionPathAsLowerCase);

  virtual ~PartitionIdGenerator() = default;

  /// Generate sequential partition IDs for input vector.
  /// @param input Input RowVector.
  /// @param result Generated integer IDs indexed by input row number.
  virtual void run(const RowVectorPtr& input, raw_vector<uint64_t>& result);

  /// Return the total number of distinct partitions processed so far.
  uint64_t numPartitions() const {
    return partitionIds_.size();
  }

  /// Return partition name for the given partition id in the typical Hive
  /// style. It is derived from the partitionValues_ at index partitionId.
  /// Partition keys appear in the order of partition columns in the table
  /// schema.
  virtual std::string partitionName(uint32_t partitionId) const;

 protected:
  PartitionIdGenerator(
      std::vector<column_index_t> partitionChannels,
      uint32_t maxPartitions,
      memory::MemoryPool* pool,
      bool partitionPathAsLowerCase);

  // Computes value IDs using VectorHashers for all rows in 'input'.
  void computeValueIds(
      const RowVectorPtr& input,
      raw_vector<uint64_t>& valueIds);

  // Computes value IDs, maps them to sequential partition IDs, and saves
  // partition values for each partition.
  // Each unique value ID gets a sequential partition ID (0, 1, 2, ...).
  // Reuses existing partition IDs for duplicate value IDs.
  // Saves partition values for new partitions for later partition name
  // generation.
  //
  // @param input RowVector containing the partition column values (either
  // original or transformed).
  // @param result Output vector to store partition IDs, indexed by row number.
  void computeAndSavePartitionIds(
      const RowVectorPtr& input,
      raw_vector<uint64_t>& result);

  const std::vector<column_index_t> partitionChannels_;

  const uint32_t maxPartitions_;

  std::vector<std::unique_ptr<exec::VectorHasher>> hashers_;

  // A vector holding unique partition key values. One row per partition. Row
  // numbers match partition IDs.
  RowVectorPtr partitionValues_;

  // A mapping from value ID produced by VectorHashers to a partition ID.
  std::unordered_map<uint64_t, uint64_t> partitionIds_;

  const bool partitionPathAsLowerCase_;

 private:
  static constexpr const int32_t kHasherReservePct = 20;

  // In case of rehash (when value IDs produced by VectorHashers change), we
  // update value id for pre-existing partitions while keeping partition ids.
  // This method rebuilds 'partitionIds_' by re-calculating the value ids using
  // updated 'hashers_'.
  void updateValueToPartitionIdMapping();

  // Copies partition values of 'row' from 'input' into 'partitionId' row in
  // 'partitionValues_'.
  virtual void savePartitionValues(
      uint32_t partitionId,
      const RowVectorPtr& input,
      vector_size_t row);

  memory::MemoryPool* const pool_;

  bool hasMultiplierSet_ = false;

  // All rows are set valid to compute partition IDs for all input rows.
  SelectivityVector allRows_;
};

} // namespace facebook::velox::connector::hive
