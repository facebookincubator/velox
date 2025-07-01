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

namespace facebook::velox::connector::hive::iceberg {
class IcebergPartitionIdGenerator : public PartitionIdGenerator {
 public:
  IcebergPartitionIdGenerator(
      const RowTypePtr& inputType,
      std::vector<column_index_t> partitionChannels,
      uint32_t maxPartitions,
      memory::MemoryPool* pool,
      ConnectorInsertTableHandlePtr insertTableHandle,
      bool partitionPathAsLowerCase)
      : PartitionIdGenerator(
            inputType,
            partitionChannels,
            maxPartitions,
            pool,
            insertTableHandle,
            partitionPathAsLowerCase) {}

  /// Generate sequential partition IDs for input vector.
  /// @param input Input RowVector.
  /// @param result Generated integer IDs indexed by input row number.
  void runIceberg(const RowVectorPtr& input, raw_vector<uint64_t>& result);
};

} // namespace facebook::velox::connector::hive::iceberg
