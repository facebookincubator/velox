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

#include "velox/exec/HashTable.h"

namespace facebook::velox::cxl {

/// Thin exec::HashTable subclass that, given a CXL pool, can relocate its
/// payload from the DRAM row container into a CXL-backed one under pressure.
template <bool ignoreNullKeys>
class CxlHashTable : public exec::HashTable<ignoreNullKeys> {
 public:
  using Base = exec::HashTable<ignoreNullKeys>;

  CxlHashTable(
      std::vector<std::unique_ptr<exec::VectorHasher>>&& hashers,
      const std::vector<exec::Accumulator>& accumulators,
      const std::vector<TypePtr>& dependentTypes,
      bool allowDuplicates,
      bool isJoinBuild,
      bool hasProbedFlag,
      bool hasCountFlag,
      uint32_t minTableSizeForParallelJoinBuild,
      memory::MemoryPool* pool,
      memory::MemoryPool* cxlPool,
      uint64_t bloomFilterMaxSize = 0);

  ~CxlHashTable() override = default;

  /// Builds an aggregation table whose payload relocates into 'cxlPool'.
  static std::unique_ptr<CxlHashTable> createForAggregation(
      std::vector<std::unique_ptr<exec::VectorHasher>>&& hashers,
      const std::vector<exec::Accumulator>& accumulators,
      memory::MemoryPool* pool,
      memory::MemoryPool* cxlPool);

  exec::RowContainer* cxlRows() const;

  /// Moves every row in the DRAM 'rows_' into 'cxlRows_' and remaps the hash
  /// bucket pointers to the new addresses.
  void relocateRowsToCxl();

 private:
  // CXL memory pool backing the relocated row container.
  memory::MemoryPool* const cxlPool_;
};

} // namespace facebook::velox::cxl
