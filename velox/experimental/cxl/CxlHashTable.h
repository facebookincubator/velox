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

  /// Like the exec::HashTable constructor; 'cxlPool', when non-null, enables
  /// relocation by building 'cxlRows_' with a layout identical to 'rows_'.
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
      uint64_t bloomFilterMaxSize = 0,
      memory::MemoryPool* cxlPool = nullptr);

  ~CxlHashTable() override = default;

  /// Builds an aggregation table. 'cxlPool', when non-null, enables relocation.
  static std::unique_ptr<CxlHashTable> createForAggregation(
      std::vector<std::unique_ptr<exec::VectorHasher>>&& hashers,
      const std::vector<exec::Accumulator>& accumulators,
      memory::MemoryPool* pool,
      memory::MemoryPool* cxlPool = nullptr) {
    return std::make_unique<CxlHashTable>(
        std::move(hashers),
        accumulators,
        std::vector<TypePtr>{},
        false, // allowDuplicates
        false, // isJoinBuild
        false, // hasProbedFlag
        false, // hasCountFlag
        0, // minTableSizeForParallelJoinBuild
        pool,
        /*bloomFilterMaxSize=*/0,
        cxlPool);
  }

  /// The CXL-backed row container, or null when no CXL pool was provided. Holds
  /// the rows relocated out of 'rows_' by relocateAllToCxl().
  exec::RowContainer* cxlRows() const {
    return cxlRows_.get();
  }

  /// Moves every row in the DRAM 'rows_' into 'cxlRows_' and swizzles the hash
  /// buckets to the new addresses, with no rehash. Requires a CXL pool.
  void relocateAllToCxl();

 protected:
  std::vector<exec::RowContainer*> additionalRows() const override {
    if (cxlRows_ == nullptr) {
      return {};
    }
    return {cxlRows_.get()};
  }

  void clear(bool freeTable) override {
    Base::clear(freeTable);
    if (cxlRows_ != nullptr) {
      cxlRows_->clear();
    }
  }

 private:
  // CXL memory pool backing 'cxlRows_'. Null when no CXL tier is configured.
  memory::MemoryPool* const cxlPool_;

  // Second row container, backed by 'cxlPool_', with a byte-identical layout to
  // 'rows_'. Holds rows relocated out of DRAM. Null when 'cxlPool_' is null.
  std::unique_ptr<exec::RowContainer> cxlRows_;
};

} // namespace facebook::velox::cxl
