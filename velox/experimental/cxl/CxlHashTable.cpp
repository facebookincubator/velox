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

#include <algorithm>
#include <cstring>

namespace facebook::velox::cxl {
namespace {

// One affine piece of a relocation: every source row in [srcBegin, srcLast]
// moved by 'delta' (new == old + delta).
struct RowRelocation {
  char* srcBegin;
  char* srcLast;
  int64_t delta;
};

// Address translator over the affine pieces returned by relocateRows. Maps a
// moved row to its new address, or returns it unchanged. Pieces are sorted by
// source address, so translate() binary-searches a handful of entries even when
// millions of rows moved.
//
// Rather than store a new address per relocated row, we exploit that rows in
// one bump-allocated run all shift by the same delta: one (srcBegin, srcLast,
// delta) piece covers the whole run, and translate() just adds the matching
// delta. So memory is O(allocation segments), not O(rows) -- measured at ~2
// pieces for 4K rows and 5 pieces for 150K rows -- which is the whole reason
// this representation is worth it.
class RowRelocationMap {
 public:
  explicit RowRelocationMap(std::vector<RowRelocation> pieces)
      : pieces_(std::move(pieces)) {}

  bool empty() const {
    return pieces_.empty();
  }

  char* translate(char* from) const {
    auto it = std::upper_bound(
        pieces_.begin(),
        pieces_.end(),
        from,
        [](char* key, const RowRelocation& piece) {
          return key < piece.srcBegin;
        });
    if (it != pieces_.begin()) {
      --it;
      if (from <= it->srcLast) {
        return from + it->delta;
      }
    }
    return from;
  }

 private:
  std::vector<RowRelocation> pieces_;
};

// Moves every row in 'source' into 'dest' with a fixed-width byte copy (row
// plus normalized-key prefix), empties 'source', and returns the move as a
// run-compressed vector of affine pieces sorted by source address. Both
// containers must share a fixed layout and hold no variable-width or external
// data. Carries 'source' per-column stats into 'dest'.
std::vector<RowRelocation> relocateRows(
    exec::RowContainer& source,
    exec::RowContainer& dest) {
  VELOX_CHECK(
      !source.usesExternalMemory() && !dest.usesExternalMemory(),
      "relocateRows supports only fixed-size rows without external memory");
  VELOX_CHECK_EQ(
      source.fixedRowSize(),
      dest.fixedRowSize(),
      "relocateRows requires an identical row layout");
  VELOX_CHECK_EQ(
      source.normalizedKeySize(),
      dest.normalizedKeySize(),
      "relocateRows requires an identical normalized-key size");
  VELOX_CHECK_EQ(
      source.nextOffset(),
      0,
      "relocateRows does not support duplicate-row links");

  // Bump-allocation in source-address order makes a run of consecutive rows map
  // to a constant delta, so record one affine piece per run, not one per row.
  // 'runStride' is the open piece's stride, 0 until it holds a second row.
  std::vector<RowRelocation> pieces;
  int64_t runStride = 0;
  // Copy the fixed row plus the normalized-key prefix in the word below it;
  // 'row - normalizedKeySize' is the start of the row's allocation.
  const auto normalizedKeySize = source.normalizedKeySize();
  const auto rowBytes =
      static_cast<size_t>(source.fixedRowSize()) + normalizedKeySize;
  constexpr int32_t kBatch = 1024;
  std::vector<char*> rows(kBatch);
  exec::RowContainerIterator iter;
  for (;;) {
    const auto numRows = source.listRows(&iter, kBatch, rows.data());
    if (numRows == 0) {
      break;
    }
    for (auto i = 0; i < numRows; ++i) {
      char* from = rows[i];
      char* to = dest.newRow();
      // A hash table stores row pointers in 48-bit bucket slots, so the
      // destination pool's addresses must fit in 48 bits.
      VELOX_CHECK_EQ(
          reinterpret_cast<uintptr_t>(to) >> 48,
          0,
          "Relocated row address does not fit in a 48-bit hash table slot");
      ::memcpy(to - normalizedKeySize, from - normalizedKeySize, rowBytes);
      // Extend the open piece if this row continues its run at the same delta
      // and stride; otherwise start a new single-row piece.
      const int64_t delta = to - from;
      if (!pieces.empty() && delta == pieces.back().delta) {
        const int64_t step = from - pieces.back().srcLast;
        if ((runStride == 0 && step > 0) ||
            (runStride != 0 && step == runStride)) {
          runStride = step;
          pieces.back().srcLast = from;
          continue;
        }
      }
      pieces.push_back({from, from, delta});
      runStride = 0;
    }
  }

  // Carry the per-column stats over to 'dest'. The byte copy preserves each
  // row's null flag, but not the aggregated stats that columnHasNulls() and the
  // null-aware extract path rely on; without this a migrated null key would be
  // read back via the no-nulls fast path as a non-null value.
  dest.mergeColumnStats(source);

  // Free 'source' row memory but keep its current normalized-key size: clear()
  // resets it to the original, which differs once a table has fallen back to
  // kHash and disabled normalized keys.
  source.clear();
  if (normalizedKeySize == 0) {
    source.disableNormalizedKeys();
  }

  // Sort by source address so the caller can binary-search; disjoint pieces are
  // thereby ordered by srcLast too.
  std::sort(pieces.begin(), pieces.end(), [](const auto& a, const auto& b) {
    return a.srcBegin < b.srcBegin;
  });
  return pieces;
}

} // namespace

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
    memory::MemoryPool* cxlPool,
    uint64_t bloomFilterMaxSize)
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
  VELOX_CHECK_NOT_NULL(cxlPool_, "CxlHashTable requires a CXL memory pool");
  // Build 'cxlRows' with the same layout as 'rows_' (same args, CXL pool). The
  // base constructor has finalized the hash mode, so the normalized-key flag
  // matches 'rows_'.
  std::vector<TypePtr> keys;
  keys.reserve(this->hashers().size());
  for (const auto& hasher : this->hashers()) {
    keys.push_back(hasher->type());
  }
  auto cxlRows = std::make_unique<exec::RowContainer>(
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
  // Hand ownership to the base table so it reindexes the container on rehash
  // and spans it in allRows()/clear(); relocateRowsToCxl() moves rows into it
  // later. The base table owns it; cxlRows() borrows it back when needed.
  this->addOtherRowContainer(std::move(cxlRows));
}

template <bool ignoreNullKeys>
std::unique_ptr<CxlHashTable<ignoreNullKeys>>
CxlHashTable<ignoreNullKeys>::createForAggregation(
    std::vector<std::unique_ptr<exec::VectorHasher>>&& hashers,
    const std::vector<exec::Accumulator>& accumulators,
    memory::MemoryPool* pool,
    memory::MemoryPool* cxlPool) {
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
      cxlPool,
      /*bloomFilterMaxSize=*/0);
}

template <bool ignoreNullKeys>
exec::RowContainer* CxlHashTable<ignoreNullKeys>::cxlRows() const {
  // The constructor always registers 'cxlRows_' as the sole other container.
  return this->otherRowContainers().front().get();
}

template <bool ignoreNullKeys>
void CxlHashTable<ignoreNullKeys>::relocateRowsToCxl() {
  auto* cxlRows = this->cxlRows();
  // Keep 'cxlRows' normalized-key state in sync with 'rows_' before the move so
  // their fixed layouts match: a table that has fallen back to kHash has
  // disabled normalized keys on 'rows_' but not on 'cxlRows'.
  if (this->hashMode() == exec::BaseHashTable::HashMode::kHash) {
    cxlRows->disableNormalizedKeys();
  }
  RowRelocationMap map(relocateRows(*this->rows(), *cxlRows));
  if (map.empty()) {
    return;
  }
  this->remapRowPointers([&map](char* from) { return map.translate(from); });
}

template class CxlHashTable<true>;
template class CxlHashTable<false>;

} // namespace facebook::velox::cxl
