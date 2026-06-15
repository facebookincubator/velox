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

#include "velox/experimental/cxl/CxlHashTable.h"

#include "velox/experimental/cxl/CxlRelocationMap.h"

namespace facebook::velox::cxl {

template <bool ignoreNullKeys>
CxlHashTable<ignoreNullKeys>::CxlHashTable(
    std::vector<std::unique_ptr<exec::VectorHasher>>&& hashers,
    const std::vector<exec::Accumulator>& accumulators,
    const std::vector<TypePtr>& dependentTypes,
    bool allowDuplicates,
    bool isJoinBuild,
    bool hasProbedFlag,
    bool hasCountFlag,
    uint32_t minTableSizeForParallelJoinBuild,
    memory::MemoryPool* pool,
    uint64_t bloomFilterMaxSize,
    memory::MemoryPool* cxlPool)
    : Base(
          std::move(hashers),
          accumulators,
          dependentTypes,
          allowDuplicates,
          isJoinBuild,
          hasProbedFlag,
          hasCountFlag,
          minTableSizeForParallelJoinBuild,
          pool,
          bloomFilterMaxSize),
      cxlPool_(cxlPool) {
  if (cxlPool_ == nullptr) {
    return;
  }
  // Build 'cxlRows_' with the same layout as 'rows_' (same args, CXL pool). The
  // base constructor has finalized the hash mode, so the normalized-key flag
  // matches 'rows_'.
  std::vector<TypePtr> keys;
  keys.reserve(this->hashers().size());
  for (const auto& hasher : this->hashers()) {
    keys.push_back(hasher->type());
  }
  cxlRows_ = std::make_unique<exec::RowContainer>(
      keys,
      !ignoreNullKeys,
      accumulators,
      dependentTypes,
      allowDuplicates,
      isJoinBuild,
      hasProbedFlag,
      hasCountFlag,
      this->hashMode() != exec::BaseHashTable::HashMode::kHash,
      /*useListRowIndex=*/false,
      cxlPool_);
}

template <bool ignoreNullKeys>
void CxlHashTable<ignoreNullKeys>::relocateAllToCxl() {
  VELOX_CHECK_NOT_NULL(
      cxlRows_, "relocateAllToCxl requires a CXL-backed row container");
  // Keep 'cxlRows_' normalized-key state in sync with 'rows_' before the move
  // so their fixed layouts match: a table that has fallen back to kHash has
  // disabled normalized keys on 'rows_' but not on 'cxlRows_'.
  if (this->hashMode() == exec::BaseHashTable::HashMode::kHash) {
    cxlRows_->disableNormalizedKeys();
  }
  RowRelocationMap map(this->rows_->relocateTo(*cxlRows_));
  if (map.empty()) {
    return;
  }
  this->swizzleRowPointers([&map](char* from) { return map.translate(from); });
}

template class CxlHashTable<true>;
template class CxlHashTable<false>;

} // namespace facebook::velox::cxl
