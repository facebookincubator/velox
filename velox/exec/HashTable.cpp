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

#include "velox/exec/HashTable.h"
#include "velox/common/base/Portability.h"
#include "velox/common/base/SimdUtil.h"
#include "velox/common/process/ProcessBase.h"
#include "velox/exec/ContainerRowSerde.h"
#include "velox/vector/VectorTypeUtils.h"

namespace facebook::velox::exec {

template <bool ignoreNullKeys>
HashTable<ignoreNullKeys>::HashTable(
    std::vector<std::unique_ptr<VectorHasher>>&& hashers,
    const std::vector<std::unique_ptr<Aggregate>>& aggregates,
    const std::vector<TypePtr>& dependentTypes,
    bool allowDuplicates,
    bool isJoinBuild,
    bool hasProbedFlag,
    memory::MappedMemory* mappedMemory)
    : BaseHashTable(std::move(hashers)),
      aggregates_(aggregates),
      isJoinBuild_(isJoinBuild) {
  std::vector<TypePtr> keys;
  for (auto& hasher : hashers_) {
    keys.push_back(hasher->type());
    if (!VectorHasher::typeKindSupportsValueIds(hasher->typeKind())) {
      hashMode_ = HashMode::kHash;
    }
  }
  rows_ = std::make_unique<RowContainer>(
      keys,
      !ignoreNullKeys,
      aggregates,
      dependentTypes,
      allowDuplicates,
      isJoinBuild,
      hasProbedFlag,
      hashMode_ != HashMode::kHash,
      mappedMemory,
      ContainerRowSerde::instance());
  nextOffset_ = rows_->nextOffset();
}

class ProbeState {
 public:
  enum class Operation { kProbe, kInsert, kErase };
  // Special tag for an erased entry. This counts as occupied for
  // probe and as empty for insert. If a tag word with empties gets an
  // erase, we make the erased tag empty. If the tag word getting the
  // erase has no empties, the erase is marked with a tombstone. A
  // probe always stops with a tag word with empties. Adding an empty
  // to a tag word with no empties would break probes that needed to
  // skip this tag word. This is standard practice for open addressing
  // hash tables. F14 has more sophistication in this but we do not
  // need it here since erase is very rare and is not expected to
  // change the load factor by much in the expected uses.
  static constexpr uint8_t kTombstoneTag = 0x7f;
  static constexpr int32_t kFullMask = 0xffff;

  int32_t row() const {
    return row_;
  }

  // Use one instruction to load 16 tags
  // Use another instruction to make 16 copies of the tag being searched for
  template <typename Table>
  inline void preProbe(const Table& table, uint64_t hash, int32_t row) {
    row_ = row;
    tagIndex_ = table.tagVectorOffset(hash);
    tagsInTable_ = BaseHashTable::loadTags(table.tags_, tagIndex_);
    auto tag = BaseHashTable::hashTag(hash);
    wantedTags_ = BaseHashTable::TagVector::broadcast(tag);
    group_ = nullptr;
    indexInTags_ = kNotSet;
    table.incrementTagLoad();
  }

  // Use one instruction to compare the tag being searched for to 16 tags
  // If there is a match, load corresponding data from the table
  template <Operation op = Operation::kProbe, typename Table>
  inline void firstProbe(const Table& table, int32_t firstKey) {
    hits_ = simd::toBitMask(tagsInTable_ == wantedTags_);
    if (hits_) {
      loadNextHit<op>(table, firstKey);
    }
  }

  template <Operation op, typename Compare, typename Insert, typename Table>
  inline char* FOLLY_NULLABLE fullProbe(
      Table& table,
      int32_t firstKey,
      Compare compare,
      Insert insert,
      bool extraCheck = false) {
    if (group_ && compare(group_, row_)) {
      if (op == Operation::kErase) {
        eraseHit(table);
      }
      table.incrementHit();
      return group_;
    }

    auto alreadyChecked = group_;
    if (extraCheck) {
      tagsInTable_ = table.loadTags(tagIndex_);
      hits_ = simd::toBitMask(tagsInTable_ == wantedTags_);
    }

    int32_t insertTagIndex = -1;
    const auto kEmptyGroup = BaseHashTable::TagVector::broadcast(0);
    for (;;) {
      if (!hits_) {
        uint16_t empty = simd::toBitMask(tagsInTable_ == kEmptyGroup);
        if (empty) {
          if (op == Operation::kProbe) {
            return nullptr;
          }
          if (op == Operation::kErase) {
            VELOX_FAIL("Erasing non-existing entry");
          }
          if (indexInTags_ != kNotSet) {
            // We came to the end of the probe without a hit. We replace the
            // first tombstone on the way.
            return insert(row_, insertTagIndex + indexInTags_);
          }
          auto pos = bits::getAndClearLastSetBit(empty);
          return insert(row_, tagIndex_ + pos);
        } else if (op == Operation::kInsert && indexInTags_ == kNotSet) {
          // We passed through a full group.
          if (table.hasTombstones_) {
            const auto kTombstoneGroup =
                BaseHashTable::TagVector::broadcast(kTombstoneTag);

            uint16_t tombstones =
                simd::toBitMask(tagsInTable_ == kTombstoneGroup);
            if (tombstones) {
              insertTagIndex = tagIndex_;
              indexInTags_ = bits::getAndClearLastSetBit(tombstones);
            }
          }
        }
      } else {
        loadNextHit<op>(table, firstKey);
        if (!(extraCheck && group_ == alreadyChecked) &&
            compare(group_, row_)) {
          if (op == Operation::kErase) {
            eraseHit(table);
          }
          table.incrementHit();
          return group_;
        }
        continue;
      }
      tagIndex_ = table.nextTagVectorOffset(tagIndex_);
      tagsInTable_ = table.loadTags(tagIndex_);
      hits_ = simd::toBitMask(tagsInTable_ == wantedTags_);
    }
  }

  template <typename Table>
  FOLLY_ALWAYS_INLINE char* FOLLY_NULLABLE
  joinNormalizedKeyFullProbe(const Table& table, const uint64_t* keys) {
    if (group_ && RowContainer::normalizedKey(group_) == keys[row_]) {
      table.incrementHit();
      return group_;
    }
    const auto kEmptyGroup = BaseHashTable::TagVector::broadcast(0);
    for (;;) {
      if (!hits_) {
        uint16_t empty = simd::toBitMask(tagsInTable_ == kEmptyGroup);
        if (empty) {
          return nullptr;
        }
      } else {
        loadNextHit<Operation::kProbe>(
            table, -static_cast<int32_t>(sizeof(normalized_key_t)));
        if (RowContainer::normalizedKey(group_) == keys[row_]) {
          table.incrementHit();
          return group_;
        }
        continue;
      }
      tagIndex_ = table.nextTagVectorOffset(tagIndex_);
      tagsInTable_ = BaseHashTable::loadTags(table.tags_, tagIndex_);
      hits_ = simd::toBitMask(tagsInTable_ == wantedTags_) & kFullMask;
    }
  }

 private:
  static constexpr uint8_t kNotSet = 0xff;

  template <Operation op, typename Table>
  inline void loadNextHit(Table& table, int32_t firstKey) {
    int32_t hit = bits::getAndClearLastSetBit(hits_);

    if (op == Operation::kErase) {
      indexInTags_ = hit;
    }
    group_ = table.row(tagIndex_, hit);
    __builtin_prefetch(group_ + firstKey);
    table.incrementRowLoad();
  }

  template <typename Table>
  void eraseHit(Table& table) {
    const auto kEmptyGroup = BaseHashTable::TagVector::broadcast(0);
    auto empty = simd::toBitMask(tagsInTable_ == kEmptyGroup);

    if (!empty) {
      table.hasTombstones_ = true;
    }
    BaseHashTable::storeTag(
        table.tags_, tagIndex_ + indexInTags_, empty ? 0 : kTombstoneTag);
  }

  char* group_;
  BaseHashTable::TagVector wantedTags_;
  BaseHashTable::TagVector tagsInTable_;
  int32_t row_;
  int32_t tagIndex_;
  BaseHashTable::MaskType hits_;

  // If op is kErase, this is the index of the current hit within the
  // group of 'tagIndex_'. If op is kInsert, this is the index of the
  // first tombstone in the group of 'insertTagIndex_'. Insert
  // replaces the first tombstone it finds. If it finds an empty
  // before finding a tombstone, it replaces the empty as soon as it
  // sees it. But the tombstone can be replaced only after finding an
  // empty and thus determining that the item being inserted is not in
  // the table.
  uint8_t indexInTags_ = kNotSet;
};

template <bool ignoreNullKeys>
void HashTable<ignoreNullKeys>::storeKeys(
    HashLookup& lookup,
    vector_size_t row) {
  for (int32_t i = 0; i < hashers_.size(); ++i) {
    auto& hasher = hashers_[i];
    rows_->store(hasher->decodedVector(), row, lookup.hits[row], i); // NOLINT
  }
}

template <bool ignoreNullKeys>
void HashTable<ignoreNullKeys>::storeRowPointer(
    int32_t index,
    uint64_t hash,
    char* row) {
  if (hashMode_ != HashMode::kArray) {
    tags_[index] = hashTag(hash);
    if (kInterleaveRows) {
      // The pointer is in slot (index - start_of_group) after the
      // tags. We first get the base address of the tag/pointer
      // group. We add the size of the tags. Then we add pointer size
      // * index of the tag in the tags vector. This is the address of
      // a 6 byte pointer to the row.
      int groupOffset = index & ~(kTagRowGroupSize - 1);
      uint64_t* pointer = reinterpret_cast<uint64_t*>(
          tags_ + sizeof(TagVector) + groupOffset +
          (kBytesInPointer * (index - groupOffset)));

      // We store 48 bits, preserving the high 16 bits of the word, which belong
      // to the next pointer.
      auto previous = *pointer & ~kPointerMask;
      *pointer = reinterpret_cast<uint64_t>(row) | previous;
      return;
    }
  }
  table_[index] = row;
}

template <bool ignoreNullKeys>
char* HashTable<ignoreNullKeys>::insertEntry(
    HashLookup& lookup,
    int32_t index,
    vector_size_t row) {
  char* group = rows_->newRow();
  lookup.hits[row] = group; // NOLINT
  storeKeys(lookup, row);
  storeRowPointer(index, lookup.hashes[row], group);
  if (hashMode_ == HashMode::kNormalizedKey) {
    // We store the unique digest of key values (normalized key) in
    // the word below the row. Space was reserved in the allocation
    // unless we have given up on normalized keys.
    RowContainer::normalizedKey(group) = lookup.normalizedKeys[row]; // NOLINT
  }
  ++numDistinct_;
  lookup.newGroups.push_back(row);
  return group;
}

template <bool ignoreNullKeys>
bool HashTable<ignoreNullKeys>::compareKeys(
    const char* group,
    HashLookup& lookup,
    vector_size_t row) {
  int32_t numKeys = lookup.hashers.size();
  // The loop runs at least once. Allow for first comparison to fail
  // before loop end check.
  int32_t i = 0;
  do {
    auto& hasher = lookup.hashers[i];
    if (!rows_->equals<!ignoreNullKeys>(
            group, rows_->columnAt(i), hasher->decodedVector(), row)) {
      return false;
    }
  } while (++i < numKeys);
  return true;
}

template <bool ignoreNullKeys>
bool HashTable<ignoreNullKeys>::compareKeys(
    const char* group,
    const char* inserted) {
  auto numKeys = hashers_.size();
  int32_t i = 0;
  do {
    if (rows_->compare(group, inserted, i, CompareFlags{true, true})) {
      return false;
    }
  } while (++i < numKeys);
  return true;
}

template <bool ignoreNullKeys>
template <bool isJoin, bool isNormalizedKey>
FOLLY_ALWAYS_INLINE void HashTable<ignoreNullKeys>::fullProbe(
    HashLookup& lookup,
    ProbeState& state,
    bool extraCheck) {
  constexpr ProbeState::Operation op =
      isJoin ? ProbeState::Operation::kProbe : ProbeState::Operation::kInsert;
  if constexpr (isNormalizedKey) {
    // NOLINT
    lookup.hits[state.row()] = state.fullProbe<op>(
        *this,
        -static_cast<int32_t>(sizeof(normalized_key_t)),
        [&](char* group, int32_t row) INLINE_LAMBDA {
          return RowContainer::normalizedKey(group) ==
              lookup.normalizedKeys[row];
        },
        [&](int32_t index, int32_t row) {
          return isJoin ? nullptr : insertEntry(lookup, row, index);
        },
        !isJoin && extraCheck);
    return;
  }
  // NOLINT
  lookup.hits[state.row()] = state.fullProbe<op>(
      *this,
      0,
      [&](char* group, int32_t row) { return compareKeys(group, lookup, row); },
      [&](int32_t index, int32_t row) {
        return isJoin ? nullptr : insertEntry(lookup, row, index);
      },
      !isJoin && extraCheck);
}

namespace {
// Normalized keys have non0-random bits. Bits need to be propagated
// up to make a tag byte and down so that non-lowest bits of
// normalized key affect the hash table index.
inline uint64_t mixNormalizedKey(uint64_t k, uint8_t bits) {
  constexpr uint64_t prime1 = 0xc6a4a7935bd1e995UL; // M from Murmurhash.
  constexpr uint64_t prime2 = 527729;
  constexpr uint64_t prime3 = 28047;
#if 1
  return folly::hasher<uint64_t>()(k);
#elif 0
  auto h = static_cast<uint64_t>(simd::crc32U64(0, k));
  auto h2 = static_cast<uint64_t>(simd::crc32U64(prime2, h + k));
  return h | (h2 << 32);
#elif 0
  auto h = k * prime1;
  return h + (k ^ (h >> 32));
#else
  auto h = (k ^ ((k >> 32))) * prime1;
  return h + (h >> bits) * prime2 + (h >> (2 * bits)) * prime3;
#endif
}

void populateNormalizedKeys(HashLookup& lookup, int8_t sizeBits) {
  lookup.normalizedKeys.resize(lookup.rows.back() + 1);
  uint64_t* __restrict hashes = lookup.hashes.data();
  uint64_t* __restrict keys = lookup.normalizedKeys.data();
  int32_t end = lookup.rows.back() + 1;
  if (end / 4 < lookup.rows.size()) {
    // For more than 1/4 of the positions in use, run the loop on all
    // elements, since the loop will do 4 at a time.
    for (auto row = 0; row < end; ++row) {
      auto hash = hashes[row];
      keys[row] = hash; // NOLINT
      hashes[row] = mixNormalizedKey(hash, sizeBits);
    }
    return;
  }
  for (auto row : lookup.rows) {
    auto hash = hashes[row];
    keys[row] = hash; // NOLINT
    hashes[row] = mixNormalizedKey(hash, sizeBits);
  }
}
} // namespace

template <bool ignoreNullKeys>
void HashTable<ignoreNullKeys>::groupProbe(HashLookup& lookup) {
  if (kTrackLoads) {
    numProbe_ += lookup.rows.size();
  }

  if (hashMode_ == HashMode::kArray) {
    arrayGroupProbe(lookup);
    return;
  }
  // Do size-based rehash before mixing hashes from normalized keys
  // because the size of the table affects the mixing.
  checkSize(lookup.rows.size());
  if (hashMode_ == HashMode::kNormalizedKey) {
    populateNormalizedKeys(lookup, sizeBits_);
    groupNormalizedKeyProbe(lookup);
    return;
  }
  ProbeState state1;
  ProbeState state2;
  ProbeState state3;
  ProbeState state4;
  int32_t probeIndex = 0;
  int32_t numProbes = lookup.rows.size();
  auto rows = lookup.rows.data();
  for (; probeIndex + 4 <= numProbes; probeIndex += 4) {
    int32_t row = rows[probeIndex];
    state1.preProbe(*this, lookup.hashes[row], row);
    row = rows[probeIndex + 1];
    state2.preProbe(*this, lookup.hashes[row], row);
    row = rows[probeIndex + 2];
    state3.preProbe(*this, lookup.hashes[row], row);
    row = rows[probeIndex + 3];
    state4.preProbe(*this, lookup.hashes[row], row);
    state1.firstProbe<ProbeState::Operation::kInsert>(*this, 0);
    state2.firstProbe<ProbeState::Operation::kInsert>(*this, 0);
    state3.firstProbe<ProbeState::Operation::kInsert>(*this, 0);
    state4.firstProbe<ProbeState::Operation::kInsert>(*this, 0);
    fullProbe<false>(lookup, state1, false);
    fullProbe<false>(lookup, state2, true);
    fullProbe<false>(lookup, state3, true);
    fullProbe<false>(lookup, state4, true);
  }
  for (; probeIndex < numProbes; ++probeIndex) {
    int32_t row = rows[probeIndex];
    state1.preProbe(*this, lookup.hashes[row], row);
    state1.firstProbe(*this, 0);
    fullProbe<false>(lookup, state1, false);
  }
  initializeNewGroups(lookup);
}

template <bool ignoreNullKeys>
void HashTable<ignoreNullKeys>::groupNormalizedKeyProbe(HashLookup& lookup) {
  ProbeState state1;
  ProbeState state2;
  ProbeState state3;
  ProbeState state4;
  int32_t probeIndex = 0;
  int32_t numProbes = lookup.rows.size();
  auto rows = lookup.rows.data();
  constexpr int32_t kKeyOffset =
      -static_cast<int32_t>(sizeof(normalized_key_t));
  for (; probeIndex + 4 <= numProbes; probeIndex += 4) {
    int32_t row = rows[probeIndex];
    state1.preProbe(*this, lookup.hashes[row], row);
    row = rows[probeIndex + 1];
    state2.preProbe(*this, lookup.hashes[row], row);
    row = rows[probeIndex + 2];
    state3.preProbe(*this, lookup.hashes[row], row);
    row = rows[probeIndex + 3];
    state4.preProbe(*this, lookup.hashes[row], row);
    state1.firstProbe<ProbeState::Operation::kInsert>(*this, kKeyOffset);
    state2.firstProbe<ProbeState::Operation::kInsert>(*this, kKeyOffset);
    state3.firstProbe<ProbeState::Operation::kInsert>(*this, kKeyOffset);
    state4.firstProbe<ProbeState::Operation::kInsert>(*this, kKeyOffset);
    fullProbe<false, true>(lookup, state1, false);
    fullProbe<false, true>(lookup, state2, true);
    fullProbe<false, true>(lookup, state3, true);
    fullProbe<false, true>(lookup, state4, true);
  }
  for (; probeIndex < numProbes; ++probeIndex) {
    int32_t row = rows[probeIndex];
    state1.preProbe(*this, lookup.hashes[row], row);
    state1.firstProbe(*this, kKeyOffset);
    fullProbe<false, true>(lookup, state1, false);
  }
  initializeNewGroups(lookup);
}

template <bool ignoreNullKeys>
void HashTable<ignoreNullKeys>::arrayGroupProbe(HashLookup& lookup) {
  VELOX_DCHECK(!lookup.hashes.empty());
  VELOX_DCHECK(!lookup.hits.empty());

  int32_t numProbes = lookup.rows.size();
  const vector_size_t* rows = lookup.rows.data();
  auto hashes = lookup.hashes.data();
  auto groups = lookup.hits.data();
  int32_t i = 0;
  if (process::hasAvx2() && simd::isDense(rows, numProbes)) {
    auto allZero = xsimd::broadcast<int64_t>(0);
    constexpr int32_t kWidth = xsimd::batch<int64_t>::size;
    auto start = rows[0];
    auto end = start + numProbes - kWidth;
    for (i = start; i <= end; i += kWidth) {
      auto loaded = simd::gather(
          reinterpret_cast<const int64_t*>(table_),
          reinterpret_cast<const int64_t*>(hashes + i));
      loaded.store_unaligned(reinterpret_cast<int64_t*>(groups + i));
      auto misses = simd::toBitMask(loaded == allZero);
      if (LIKELY(!misses)) {
        continue;
      }
      for (auto miss = 0; miss < kWidth; ++miss) {
        auto row = i + miss;
        if (!groups[row]) {
          auto index = hashes[row];
          auto hit = table_[index];
          if (!hit) {
            hit = insertEntry(lookup, index, row);
          }
          groups[row] = hit;
        }
      }
    }
    i -= start;
  }
  for (; i < numProbes; ++i) {
    auto row = rows[i];
    uint64_t index = hashes[row];
    VELOX_DCHECK(index < size_);
    char* group = table_[index];
    if (UNLIKELY(!group)) {
      group = insertEntry(lookup, index, row);
    }
    groups[row] = group;
    lookup.hits[row] = group; // NOLINT
  }
  initializeNewGroups(lookup);
}

template <bool ignoreNullKeys>
void HashTable<ignoreNullKeys>::joinProbe(HashLookup& lookup) {
  if (kTrackLoads) {
    numProbe_ += lookup.rows.size();
  }
  if (hashMode_ == HashMode::kArray) {
    arrayJoinProbe(lookup);
    return;
  }
  if (hashMode_ == HashMode::kNormalizedKey) {
    populateNormalizedKeys(lookup, sizeBits_);
    joinNormalizedKeyProbe(lookup);
    return;
  }
  int32_t probeIndex = 0;
  int32_t numProbes = lookup.rows.size();
  const vector_size_t* rows = lookup.rows.data();
  ProbeState state1;
  ProbeState state2;
  ProbeState state3;
  ProbeState state4;
  for (; probeIndex + 4 <= numProbes; probeIndex += 4) {
    int32_t row = rows[probeIndex];
    state1.preProbe(*this, lookup.hashes[row], row);
    row = rows[probeIndex + 1];
    state2.preProbe(*this, lookup.hashes[row], row);
    row = rows[probeIndex + 2];
    state3.preProbe(*this, lookup.hashes[row], row);
    row = rows[probeIndex + 3];
    state4.preProbe(*this, lookup.hashes[row], row);
    state1.firstProbe(*this, 0);
    state2.firstProbe(*this, 0);
    state3.firstProbe(*this, 0);
    state4.firstProbe(*this, 0);
    fullProbe<true>(lookup, state1, false);
    fullProbe<true>(lookup, state2, false);
    fullProbe<true>(lookup, state3, false);
    fullProbe<true>(lookup, state4, false);
  }
  for (; probeIndex < numProbes; ++probeIndex) {
    int32_t row = rows[probeIndex];
    state1.preProbe(*this, lookup.hashes[row], row);
    state1.firstProbe(*this, 0);
    fullProbe<true>(lookup, state1, false);
  }
}

template <bool ignoreNullKeys>
void HashTable<ignoreNullKeys>::arrayJoinProbe(HashLookup& lookup) {
  // Rows are nearly always consecutive.
  auto& rows = lookup.rows;
  auto hashes = lookup.hashes.data();
  auto hits = lookup.hits.data();
  auto numRows = rows.size();
  int32_t i = 0;
  for (; i + 8 <= numRows; i += 8) {
    auto row = rows[i];
    if (rows[i + 7] - row == 7) {
      // 8 consecutive.
      simd::gather(
          reinterpret_cast<const int64_t*>(table_),
          reinterpret_cast<const int64_t*>(hashes + row))
          .store_unaligned(reinterpret_cast<int64_t*>(hits) + row);
      simd::gather(
          reinterpret_cast<const int64_t*>(table_),
          reinterpret_cast<const int64_t*>(hashes + row + 4))
          .store_unaligned(reinterpret_cast<int64_t*>(hits) + row + 4);
    } else {
      for (auto j = i; j < i + 8; ++j) {
        auto row = rows[j];
        auto index = hashes[row];
        DCHECK(index < size_);
        hits[row] = table_[index]; // NOLINT
      }
    }
  }
  for (; i < numRows; ++i) {
    auto row = rows[i];
    auto index = hashes[row];
    DCHECK(index < size_);
    hits[row] = table_[index]; // NOLINT
  }
}

template <bool ignoreNullKeys>
void HashTable<ignoreNullKeys>::joinNormalizedKeyProbe(HashLookup& lookup) {
  int32_t probeIndex = 0;
  int32_t numProbes = lookup.rows.size();
  const vector_size_t* rows = lookup.rows.data();
  ProbeState state1;
  ProbeState state2;
  ProbeState state3;
  ProbeState state4;
  const uint64_t* keys = lookup.normalizedKeys.data();
  const uint64_t* hashes = lookup.hashes.data();
  char** hits = lookup.hits.data();
  constexpr int32_t kKeyOffset =
      -static_cast<int32_t>(sizeof(normalized_key_t));
  for (; probeIndex + 4 <= numProbes; probeIndex += 4) {
    int32_t row = rows[probeIndex];
    state1.preProbe(*this, hashes[row], row);
    row = rows[probeIndex + 1];
    state2.preProbe(*this, hashes[row], row);
    row = rows[probeIndex + 2];
    state3.preProbe(*this, hashes[row], row);
    row = rows[probeIndex + 3];
    state4.preProbe(*this, hashes[row], row);
    state1.firstProbe(*this, kKeyOffset);
    state2.firstProbe(*this, kKeyOffset);
    state3.firstProbe(*this, kKeyOffset);
    state4.firstProbe(*this, kKeyOffset);
    hits[state1.row()] = state1.joinNormalizedKeyFullProbe(*this, keys);
    hits[state2.row()] = state2.joinNormalizedKeyFullProbe(*this, keys);
    hits[state3.row()] = state3.joinNormalizedKeyFullProbe(*this, keys);
    hits[state4.row()] = state4.joinNormalizedKeyFullProbe(*this, keys);
  }
  for (; probeIndex < numProbes; ++probeIndex) {
    int32_t row = rows[probeIndex];
    state1.preProbe(*this, lookup.hashes[row], row);
    state1.firstProbe(*this, 0);
    hits[row] = state1.joinNormalizedKeyFullProbe(*this, keys);
  }
}

template <bool ignoreNullKeys>
void HashTable<ignoreNullKeys>::initializeNewGroups(HashLookup& lookup) {
  if (lookup.newGroups.empty()) {
    return;
  }
  for (auto& aggregate : aggregates_) {
    aggregate->initializeNewGroups(lookup.hits.data(), lookup.newGroups);
  }
}

template <bool ignoreNullKeys>
void HashTable<ignoreNullKeys>::allocateTables(uint64_t size) {
  VELOX_CHECK(bits::isPowerOfTwo(size), "Size is not a power of two: {}", size);
  if (size > 0) {
    setSize(size);
    constexpr auto kPageSize = memory::MappedMemory::kPageSize;
    if (kInterleaveRows) {
      // The total size is 8 bytes per slot, in groups of 16 slots
      // with 16 bytes of tags and 16 * 6 bytes of pointers and a
      // padding of 16 bytes to round up the cache line.
      auto numPages =
          bits::roundUp(size * sizeof(char*), kPageSize) / kPageSize;
      if (!rows_->mappedMemory()->allocateContiguous(
              numPages, nullptr, tableAllocation_)) {
        VELOX_FAIL("Could not allocate join/group by hash table");
      }
      tags_ = tableAllocation_.data<uint8_t>();
      table_ = nullptr;
      memset(tags_, 0, size_ * sizeof(char*));
    } else {
      // The total size is 9 bytes per slot, 8 in the pointers table and 1 in
      // the tags table.
      auto numPages = bits::roundUp(size * 9, kPageSize) / kPageSize;
      if (!rows_->mappedMemory()->allocateContiguous(
              numPages, nullptr, tableAllocation_)) {
        VELOX_FAIL("Could not allocate join/group by hash table");
      }
      table_ = tableAllocation_.data<char*>();
      tags_ = reinterpret_cast<uint8_t*>(table_ + size);
      memset(tags_, 0, size_);
      // Not strictly necessary to clear 'table_' but more debuggable.
      memset(table_, 0, size_ * sizeof(char*));
    }
  }
}

template <bool ignoreNullKeys>
void HashTable<ignoreNullKeys>::clear() {
  rows_->clear();
  if (hashMode_ != HashMode::kArray && tags_) {
    memset(tags_, 0, kInterleaveRows ? size_ * sizeof(char*) : size_);
  }
  if (table_) {
    memset(table_, 0, sizeof(char*) * size_);
  }
  numDistinct_ = 0;
}

template <bool ignoreNullKeys>
void HashTable<ignoreNullKeys>::checkSize(int32_t numNew) {
  if (!tags_ || !size_) {
    // Initial guess of cardinality is double the first input batch or at
    // least 2K entries.
    // numDistinct_ is non-0 when switching from HashMode::kArray to regular
    // hashing.
    auto newSize = std::max(
        (uint64_t)2048, bits::nextPowerOfTwo(numNew * 2 + numDistinct_));
    allocateTables(newSize);
    if (numDistinct_) {
      rehash();
    }
  } else if (numNew + numDistinct_ > rehashSize()) {
    auto newSize = bits::nextPowerOfTwo(size_ + numNew);
    allocateTables(newSize);
    rehash();
  }
}

template <bool ignoreNullKeys>
bool HashTable<ignoreNullKeys>::insertBatch(
    char** groups,
    int32_t numGroups,
    raw_vector<uint64_t>& hashes) {
  for (int32_t i = 0; i < hashers_.size(); ++i) {
    auto& hasher = hashers_[i];
    if (hashMode_ == HashMode::kHash) {
      rows_->hash(
          i, folly::Range<char**>(groups, numGroups), i > 0, hashes.data());
    } else {
      // Array or normalized key.
      auto column = rows_->columnAt(i);
      if (!hasher->computeValueIdsForRows(
              groups,
              numGroups,
              column.offset(),
              column.nullByte(),
              ignoreNullKeys ? 0 : column.nullMask(),
              hashes)) {
        // Must reconsider 'hashMode_' and start over.
        return false;
      }
    }
  }
  if (isJoinBuild_) {
    insertForJoin(groups, hashes.data(), numGroups);
  } else {
    insertForGroupBy(groups, hashes.data(), numGroups);
  }
  return true;
}

template <bool ignoreNullKeys>
void HashTable<ignoreNullKeys>::insertForGroupBy(
    char** groups,
    uint64_t* hashes,
    int32_t numGroups) {
  if (hashMode_ == HashMode::kArray) {
    for (auto i = 0; i < numGroups; ++i) {
      auto index = hashes[i];
      VELOX_CHECK_LT(index, size_);
      VELOX_CHECK_NULL(table_[index]);
      table_[index] = groups[i];
    }
  } else {
    if (hashMode_ == HashMode::kNormalizedKey) {
      for (int i = 0; i < numGroups; ++i) {
        auto hash = hashes[i];
        // Write the normalized key below the row.
        RowContainer::normalizedKey(groups[i]) = hash;
        // Shuffle the bits im the normalized key.
        hashes[i] = mixNormalizedKey(hash, sizeBits_);
      }
    }
    constexpr int32_t kBatchSize = 16;
    for (int32_t base = 0; base < numGroups; base += kBatchSize) {
      char** pointers[kBatchSize];
      auto batchEnd = std::min(base + kBatchSize, numGroups);
      for (auto j = base; j < batchEnd; ++j) {
        auto hash = hashes[j];
        auto tagIndex = tagVectorOffset(hash);
        auto tagsInTable = BaseHashTable::loadTags(tags_, tagIndex);
        MaskType free = ~simd::toBitMask(
            BaseHashTable::TagVector::batch_bool_type(tagsInTable));
        if (free) {
          int freeOffset = __builtin_ctz(free);
          tags_[tagIndex + freeOffset] = BaseHashTable::hashTag(hash);
          char** pointer;
          if (kInterleaveRows) {
            pointer = reinterpret_cast<char**>(
                tags_ + tagIndex + sizeof(TagVector) +
                kBytesInPointer * freeOffset);
          } else {
            pointer = table_ + tagIndex + freeOffset;
          }
          __builtin_prefetch(pointer);
          pointers[j - base] = pointer;
        } else {
          pointers[j - base] = nullptr;
        }
      }
      for (auto j = base; j < batchEnd; ++j) {
        auto pointer = pointers[j - base];
        if (pointer) {
          if (kInterleaveRows) {
            auto previous =
                reinterpret_cast<uint64_t>(*pointer) & ~kPointerMask;
            *pointer = reinterpret_cast<char*>(
                previous | reinterpret_cast<uint64_t>(groups[j]));
          } else {
            *pointer = groups[j];
          }
          continue;
        }
        // The place in the table was not determined on the first loop. Loop
        // until finding a free tag.
        auto hash = hashes[j] +
            (kInterleaveRows ? kTagRowGroupSize : sizeof(TagVector));
        auto tagIndex = tagVectorOffset(hash);
        auto tagsInTable = BaseHashTable::loadTags(tags_, tagIndex);
        for (;;) {
          MaskType free = ~simd::toBitMask(
              BaseHashTable::TagVector::batch_bool_type(tagsInTable));
          if (free) {
            auto freeOffset = bits::getAndClearLastSetBit(free);
            storeRowPointer(tagIndex + freeOffset, hash, groups[j]);
            break;
          }
          tagIndex = nextTagVectorOffset(tagIndex);
          tagsInTable = loadTags(tagIndex);
        }
      }
    }
  }
}

template <bool ignoreNullKeys>
bool HashTable<ignoreNullKeys>::arrayPushRow(char* row, int32_t index) {
  auto existing = table_[index];
  if (nextOffset_) {
    nextRow(row) = existing;
    if (existing) {
      hasDuplicates_ = true;
    }
  } else if (existing) {
    // Semijoin or a known unique build side ignores a repeat of a key.
    return false;
  }
  table_[index] = row;
  return !existing;
}

template <bool ignoreNullKeys>
void HashTable<ignoreNullKeys>::pushNext(char* row, char* next) {
  if (nextOffset_) {
    hasDuplicates_ = true;
    auto previousNext = nextRow(row);
    nextRow(row) = next;
    nextRow(next) = previousNext;
  }
}

template <bool ignoreNullKeys>
FOLLY_ALWAYS_INLINE void HashTable<ignoreNullKeys>::buildFullProbe(
    ProbeState& state,
    uint64_t hash,
    char* inserted,
    bool extraCheck) {
  if (hashMode_ == HashMode::kNormalizedKey) {
    state.fullProbe<ProbeState::Operation::kInsert>(
        *this,
        -static_cast<int32_t>(sizeof(normalized_key_t)),
        [&](char* group, int32_t /*row*/) {
          if (RowContainer::normalizedKey(group) ==
              RowContainer::normalizedKey(inserted)) {
            if (nextOffset_) {
              pushNext(group, inserted);
            }
            return true;
          }
          return false;
        },
        [&](int32_t /*row*/, int32_t index) {
          storeRowPointer(index, hash, inserted);
          return nullptr;
        },
        extraCheck);
  } else {
    state.fullProbe<ProbeState::Operation::kInsert>(
        *this,
        0,
        [&](char* group, int32_t /*row*/) {
          if (compareKeys(group, inserted)) {
            if (nextOffset_) {
              pushNext(group, inserted);
            }
            return true;
          }
          return false;
        },
        [&](int32_t /*row*/, int32_t index) {
          storeRowPointer(index, hash, inserted);
          return nullptr;
        },
        extraCheck);
  }
}

template <bool ignoreNullKeys>
void HashTable<ignoreNullKeys>::insertForJoin(
    char** groups,
    uint64_t* hashes,
    int32_t numGroups) {
  if (hashMode_ == HashMode::kNormalizedKey) {
    // Write the normalized key below each row. The key is only known
    // at the time of insert, so cannot be filled in at the time of
    // accumulating the build rows.
    for (auto i = 0; i < numGroups; ++i) {
      RowContainer::normalizedKey(groups[i]) = hashes[i];
      hashes[i] = mixNormalizedKey(hashes[i], sizeBits_);
    }
  }
  // The insertable rows are in the table, all get put in the hash
  // table or array.
  if (hashMode_ == HashMode::kArray) {
    for (auto i = 0; i < numGroups; ++i) {
      auto index = hashes[i];
      VELOX_CHECK_LT(index, size_);
      arrayPushRow(groups[i], index);
    }
    return;
  }

  ProbeState state1;
  for (auto i = 0; i < numGroups; ++i) {
    state1.preProbe(*this, hashes[i], i);
    state1.firstProbe(*this, 0);
    buildFullProbe(state1, hashes[i], groups[i], i);
  }
}

template <bool ignoreNullKeys>
void HashTable<ignoreNullKeys>::rehash() {
  constexpr int32_t kHashBatchSize = 1024;
  // @lint-ignore CLANGTIDY
  raw_vector<uint64_t> hashes;
  hashes.resize(kHashBatchSize);
  char* groups[kHashBatchSize];
  // A join build can have multiple payload tables. Loop over 'this'
  // and the possible other tables and put all the data in the table
  // of 'this'.
  for (int32_t i = 0; i <= otherTables_.size(); ++i) {
    RowContainerIterator iterator;
    int32_t numGroups;
    do {
      numGroups = (i == 0 ? this : otherTables_[i - 1].get())
                      ->rows()
                      ->listRows(&iterator, kHashBatchSize, groups);
      if (!insertBatch(groups, numGroups, hashes)) {
        VELOX_CHECK(hashMode_ != HashMode::kHash);
        setHashMode(HashMode::kHash, 0);
        return;
      }
    } while (numGroups > 0);
  }
}

template <bool ignoreNullKeys>
void HashTable<ignoreNullKeys>::setHashMode(HashMode mode, int32_t numNew) {
  VELOX_CHECK(hashMode_ != HashMode::kHash);
  if (mode == HashMode::kArray) {
    auto bytes = size_ * sizeof(char*);
    constexpr auto kPageSize = memory::MappedMemory::kPageSize;
    auto numPages = bits::roundUp(bytes, kPageSize) / kPageSize;
    if (!rows_->mappedMemory()->allocateContiguous(
            numPages, nullptr, tableAllocation_)) {
      VELOX_FAIL("Could not allocate array for array mode hash table");
    }
    table_ = tableAllocation_.data<char*>();
    memset(table_, 0, bytes);
    hashMode_ = HashMode::kArray;
    rehash();
  } else if (mode == HashMode::kHash) {
    hashMode_ = HashMode::kHash;
    for (auto& hasher : hashers_) {
      hasher->resetStats();
    }
    rows_->disableNormalizedKeys();
    size_ = 0;
    // Makes tables of the right size and rehashes.
    checkSize(numNew);
  } else if (mode == HashMode::kNormalizedKey) {
    hashMode_ = HashMode::kNormalizedKey;
    size_ = 0;
    // Makes tables of the right size and rehashes.
    checkSize(numNew);
  }
}

template <bool ignoreNullKeys>
bool HashTable<ignoreNullKeys>::analyze() {
  constexpr int32_t kHashBatchSize = 1024;
  // @lint-ignore CLANGTIDY
  char* groups[kHashBatchSize];
  RowContainerIterator iterator;
  int32_t numGroups;
  do {
    numGroups = rows_->listRows(&iterator, kHashBatchSize, groups);
    for (int32_t i = 0; i < hashers_.size(); ++i) {
      auto& hasher = hashers_[i];
      if (!hasher->isRange()) {
        // A range mode hasher does not know distincts, so need to
        // look. A distinct mode one does know the range. A hash join
        // build is always analyzed.
        continue;
      }
      uint64_t rangeSize;
      uint64_t distinctSize;
      hasher->cardinality(0, rangeSize, distinctSize);
      if (distinctSize == VectorHasher::kRangeTooLarge &&
          rangeSize == VectorHasher::kRangeTooLarge) {
        return false;
      }
      RowColumn column = rows_->columnAt(i);
      hasher->analyze(
          groups,
          numGroups,
          column.offset(),
          ignoreNullKeys ? 0 : column.nullByte(),
          ignoreNullKeys ? 0 : column.nullMask());
    }
  } while (numGroups > 0);
  return true;
}

namespace {
// Multiplies a * b and produces uint64_t max to denote overflow. If
// either a or b is overflow, preserves overflow.
inline uint64_t safeMul(uint64_t a, uint64_t b) {
  constexpr uint64_t kMax = std::numeric_limits<uint64_t>::max();
  if (a == kMax || b == kMax) {
    return kMax;
  }
  uint64_t result;
  if (__builtin_mul_overflow(a, b, &result)) {
    return kMax;
  }
  return result;
}
} // namespace

template <bool ignoreNullKeys>
void HashTable<ignoreNullKeys>::enableRangeWhereCan(
    const std::vector<uint64_t>& rangeSizes,
    const std::vector<uint64_t>& distinctSizes,
    std::vector<bool>& useRange) {
  // Sort non-range keys by the cardinality increase going from distinct to
  // range.
  std::vector<size_t> indices(rangeSizes.size());
  std::vector<uint64_t> rangeMultipliers(
      rangeSizes.size(), std::numeric_limits<uint64_t>::max());
  for (auto i = 0; i < rangeSizes.size(); i++) {
    indices[i] = i;
    if (!useRange[i]) {
      rangeMultipliers[i] = rangeSizes[i] / distinctSizes[i];
    }
  }

  std::sort(indices.begin(), indices.end(), [&](auto i, auto j) {
    return rangeMultipliers[i] < rangeMultipliers[j];
  });

  auto calculateNewMultipler = [&]() {
    uint64_t multipler = 1;
    for (auto i = 0; i < rangeSizes.size(); ++i) {
      auto kind = hashers_[i]->typeKind();
      // NOLINT
      multipler =
          safeMul(multipler, useRange[i] ? rangeSizes[i] : distinctSizes[i]);
    }
    return multipler;
  };

  // Switch distinct to range if the cardinality increase does not overflow
  // 64 bits.
  for (auto i = 0; i < rangeSizes.size(); ++i) {
    if (!useRange[indices[i]]) {
      useRange[indices[i]] = true;
      auto newProduct = calculateNewMultipler();
      if (newProduct == VectorHasher::kRangeTooLarge) {
        useRange[indices[i]] = false;
        return;
      }
    }
  }
}

template <bool ignoreNullKeys>
uint64_t HashTable<ignoreNullKeys>::setHasherMode(
    const std::vector<std::unique_ptr<VectorHasher>>& hashers,
    const std::vector<bool>& useRange,
    const std::vector<uint64_t>& rangeSizes,
    const std::vector<uint64_t>& distinctSizes) {
  uint64_t multiplier = 1;
  // A group by leaves 50% space for values not yet seen.
  for (int i = 0; i < hashers.size(); ++i) {
    auto kind = hashers[i]->typeKind();
    multiplier = useRange.size() > i && useRange[i]
        ? hashers[i]->enableValueRange(multiplier, reservePct())
        : hashers[i]->enableValueIds(multiplier, reservePct());
    VELOX_CHECK_NE(multiplier, VectorHasher::kRangeTooLarge);
  }
  return multiplier;
}

template <bool ignoreNullKeys>
void HashTable<ignoreNullKeys>::clearUseRange(std::vector<bool>& useRange) {
  for (auto i = 0; i < hashers_.size(); ++i) {
    useRange[i] = hashers_[i]->typeKind() == TypeKind::BOOLEAN;
  }
}

template <bool ignoreNullKeys>
void HashTable<ignoreNullKeys>::decideHashMode(int32_t numNew) {
  std::vector<uint64_t> rangeSizes(hashers_.size());
  std::vector<uint64_t> distinctSizes(hashers_.size());
  std::vector<bool> useRange(hashers_.size());
  uint64_t bestWithReserve = 1;
  uint64_t distinctsWithReserve = 1;
  uint64_t rangesWithReserve = 1;
  if (numDistinct_ && !isJoinBuild_) {
    if (!analyze()) {
      setHashMode(HashMode::kHash, numNew);
      return;
    }
  }
  for (int i = 0; i < hashers_.size(); ++i) {
    auto kind = hashers_[i]->typeKind();
    hashers_[i]->cardinality(reservePct(), rangeSizes[i], distinctSizes[i]);
    distinctsWithReserve = safeMul(distinctsWithReserve, distinctSizes[i]);
    rangesWithReserve = safeMul(rangesWithReserve, rangeSizes[i]);
    if (distinctSizes[i] == VectorHasher::kRangeTooLarge &&
        rangeSizes[i] != VectorHasher::kRangeTooLarge) {
      useRange[i] = true;
      bestWithReserve = safeMul(bestWithReserve, rangeSizes[i]);
    } else if (
        rangeSizes[i] != VectorHasher::kRangeTooLarge &&
        rangeSizes[i] <= distinctSizes[i] * 20) {
      useRange[i] = true;
      bestWithReserve = safeMul(bestWithReserve, rangeSizes[i]);
    } else {
      bestWithReserve = safeMul(bestWithReserve, distinctSizes[i]);
    }
  }

  if (rangesWithReserve < kArrayHashMaxSize) {
    std::fill(useRange.begin(), useRange.end(), true);
    size_ = setHasherMode(hashers_, useRange, rangeSizes, distinctSizes);
    setHashMode(HashMode::kArray, numNew);
    return;
  }

  if (bestWithReserve < kArrayHashMaxSize) {
    size_ = setHasherMode(hashers_, useRange, rangeSizes, distinctSizes);
    setHashMode(HashMode::kArray, numNew);
    return;
  }
  if (rangesWithReserve != VectorHasher::kRangeTooLarge) {
    std::fill(useRange.begin(), useRange.end(), true);
    setHasherMode(hashers_, useRange, rangeSizes, distinctSizes);
    setHashMode(HashMode::kNormalizedKey, numNew);
    return;
  }
  if (hashers_.size() == 1 && distinctsWithReserve > 10000) {
    // A single part group by that does not go by range or become an array
    // does not make sense as a normalized key unless it is very small.
    setHashMode(HashMode::kHash, numNew);
    return;
  }

  if (distinctsWithReserve < kArrayHashMaxSize) {
    clearUseRange(useRange);
    size_ = setHasherMode(hashers_, useRange, rangeSizes, distinctSizes);
    setHashMode(HashMode::kArray, numNew);
    return;
  }
  if (distinctsWithReserve == VectorHasher::kRangeTooLarge &&
      rangesWithReserve == VectorHasher::kRangeTooLarge) {
    setHashMode(HashMode::kHash, numNew);
    return;
  }
  // The key concatenation fits in 64 bits.
  if (bestWithReserve != VectorHasher::kRangeTooLarge) {
    enableRangeWhereCan(rangeSizes, distinctSizes, useRange);
    setHasherMode(hashers_, useRange, rangeSizes, distinctSizes);
  } else {
    clearUseRange(useRange);
  }
  setHashMode(HashMode::kNormalizedKey, numNew);
}

template <bool ignoreNullKeys>
std::string HashTable<ignoreNullKeys>::toString() {
  std::stringstream out;
  int32_t occupied = 0;
  if (table_ && tableAllocation_.data() && tableAllocation_.size()) {
    // 'size_' and 'table_' may not be set if initializing.
    uint64_t size =
        std::min<uint64_t>(tableAllocation_.size() / sizeof(char*), size_);
    for (int32_t i = 0; i < size; ++i) {
      occupied += table_[i] != nullptr;
    }
  }
  out << "[HashTable  size: " << size_ << " occupied: " << occupied << "]";
  if (!table_) {
    out << "(no table) ";
  }
  for (auto& hasher : hashers_) {
    out << hasher->toString();
  }
  if (kTrackLoads) {
    out << std::endl;
    out << fmt::format(
        "{} probes {} tag loads {} row loads {} hits",
        numProbe_,
        numTagLoad_,
        numRowLoad_,
        numHit_);
  }
  if (hashMode_ != HashMode::kArray) {
    // Count of groups indexed by number of non-empty slots.
    int32_t numGroups[sizeof(TagVector) + 1] = {};
    int32_t tagIndex = 0;
    for (auto i = 0; i < size_; i += sizeof(TagVector)) {
      auto tags = loadTags(tagIndex);
      auto filled = simd::toBitMask(tags != TagVector::broadcast(0));
      ++numGroups[__builtin_popcount(filled)];
      tagIndex = nextTagVectorOffset(tagIndex);
    }
    out << std::endl;
    for (auto i = 0; i < sizeof(numGroups) / sizeof(numGroups[0]); ++i) {
      out << numGroups[i] << " groups with " << i << " entries" << std::endl;
    }
  }
  return out.str();
}

namespace {
bool mayUseValueIds(const BaseHashTable& table) {
  if (table.hashMode() == BaseHashTable::HashMode::kHash) {
    return false;
  }
  for (auto& hasher : table.hashers()) {
    if (!hasher->mayUseValueIds()) {
      return false;
    }
  }
  return true;
}
} // namespace

template <bool ignoreNullKeys>
void HashTable<ignoreNullKeys>::prepareJoinTable(
    std::vector<std::unique_ptr<BaseHashTable>> tables) {
  otherTables_.reserve(tables.size());
  for (auto& table : tables) {
    otherTables_.emplace_back(std::unique_ptr<HashTable<ignoreNullKeys>>(
        dynamic_cast<HashTable<ignoreNullKeys>*>(table.release())));
  }
  bool useValueIds = mayUseValueIds(*this);
  if (useValueIds) {
    for (auto& other : otherTables_) {
      if (!mayUseValueIds(*other)) {
        useValueIds = false;
        break;
      }
    }
    if (useValueIds) {
      for (auto& other : otherTables_) {
        for (auto i = 0; i < hashers_.size(); ++i) {
          hashers_[i]->merge(*other->hashers_[i]);
          if (!hashers_[i]->mayUseValueIds()) {
            useValueIds = false;
            break;
          }
        }
        if (!useValueIds) {
          break;
        }
      }
    }
  }
  numDistinct_ = rows()->numRows();
  for (auto& other : otherTables_) {
    numDistinct_ += other->rows()->numRows();
  }
  if (!useValueIds) {
    if (hashMode_ != HashMode::kHash) {
      setHashMode(HashMode::kHash, 0);
    } else {
      checkSize(0);
    }
  } else {
    decideHashMode(0);
  }
}

template <bool ignoreNullKeys>
int32_t HashTable<ignoreNullKeys>::listJoinResults(
    JoinResultIterator& iter,
    bool includeMisses,
    folly::Range<vector_size_t*> inputRows,
    folly::Range<char**> hits) {
  VELOX_CHECK_LE(inputRows.size(), hits.size());
  if (!hasDuplicates_) {
    return listJoinResultsNoDuplicates(iter, includeMisses, inputRows, hits);
  }
  int numOut = 0;
  auto maxOut = inputRows.size();
  while (iter.lastRowIndex < iter.rows->size()) {
    if (!iter.nextHit) {
      auto row = (*iter.rows)[iter.lastRowIndex];
      iter.nextHit = (*iter.hits)[row]; // NOLINT
      if (!iter.nextHit) {
        ++iter.lastRowIndex;

        if (includeMisses) {
          inputRows[numOut] = row; // NOLINT
          hits[numOut] = nullptr;
          ++numOut;
          if (numOut >= maxOut) {
            return numOut;
          }
        }
        continue;
      }
    }
    while (iter.nextHit) {
      char* next = nullptr;
      if (nextOffset_) {
        next = nextRow(iter.nextHit);
        if (next) {
          __builtin_prefetch(reinterpret_cast<char*>(next) + nextOffset_);
        }
      }
      inputRows[numOut] = (*iter.rows)[iter.lastRowIndex]; // NOLINT
      hits[numOut] = iter.nextHit;
      ++numOut;
      iter.nextHit = next;
      if (!iter.nextHit) {
        ++iter.lastRowIndex;
      }
      if (numOut >= maxOut) {
        return numOut;
      }
    }
  }
  return numOut;
}

template <bool ignoreNullKeys>
int32_t HashTable<ignoreNullKeys>::listJoinResultsNoDuplicates(
    JoinResultIterator& iter,
    bool includeMisses,
    folly::Range<vector_size_t*> inputRows,
    folly::Range<char**> hits) {
  int32_t numOut = 0;
  auto maxOut = inputRows.size();
  int32_t i = iter.lastRowIndex;
  auto numRows = iter.rows->size();

  constexpr int32_t kWidth = xsimd::batch<int64_t>::size;
  auto sourceHits = reinterpret_cast<int64_t*>(iter.hits->data());
  auto sourceRows = iter.rows->data();
  // We pass the pointers as int64_t's in 'hitWords'.
  auto resultHits = reinterpret_cast<int64_t*>(hits.data());
  auto resultRows = inputRows.data();
  int32_t outLimit = maxOut - kWidth;
  for (; i + kWidth <= numRows && numOut < outLimit; i += kWidth) {
    auto indices = simd::loadGatherIndices<int64_t, int32_t>(sourceRows + i);
    auto hitWords = simd::gather(sourceHits, indices);
    auto misses = includeMisses ? 0 : simd::toBitMask(hitWords == 0);
    if (misses == 0xf) {
      continue;
    }
    if (!misses) {
      hitWords.store_unaligned(resultHits + numOut);
      indices.store_unaligned(resultRows + numOut);
      numOut += kWidth;
      continue;
    }
    auto matches = misses ^ bits::lowMask(kWidth);
    simd::filter<int64_t>(hitWords, matches, xsimd::default_arch{})
        .store_unaligned(resultHits + numOut);
    simd::filter<int32_t>(indices, matches, xsimd::default_arch{})
        .store_unaligned(resultRows + numOut);
    numOut += __builtin_popcount(matches);
  }
  for (; i < numRows; ++i) {
    auto row = sourceRows[i];
    if (includeMisses || sourceHits[row]) {
      resultHits[numOut] = sourceHits[row];
      resultRows[numOut] = row;
      ++numOut;
      if (numOut >= maxOut) {
        ++i;
        break;
      }
    }
  }

  iter.lastRowIndex = i;
  return numOut;
}

template <bool ignoreNullKeys>
template <RowContainer::ProbeType probeType>
int32_t HashTable<ignoreNullKeys>::listRows(
    RowsIterator* iter,
    int32_t maxRows,
    uint64_t maxBytes,
    char** rows) {
  if (iter->hashTableIndex_ == -1) {
    auto numRows = rows_->listRows<probeType>(
        &iter->rowContainerIterator_, maxRows, maxBytes, rows);
    if (numRows) {
      return numRows;
    }
    iter->hashTableIndex_ = 0;
    iter->rowContainerIterator_.reset();
  }
  while (iter->hashTableIndex_ < otherTables_.size()) {
    auto numRows =
        otherTables_[iter->hashTableIndex_]
            ->rows()
            ->template listRows<probeType>(
                &iter->rowContainerIterator_, maxRows, maxBytes, rows);
    if (numRows) {
      return numRows;
    }
    ++iter->hashTableIndex_;
    iter->rowContainerIterator_.reset();
  }

  return 0;
}

template <bool ignoreNullKeys>
int32_t HashTable<ignoreNullKeys>::listNotProbedRows(
    RowsIterator* iter,
    int32_t maxRows,
    uint64_t maxBytes,
    char** rows) {
  return listRows<RowContainer::ProbeType::kNotProbed>(
      iter, maxRows, maxBytes, rows);
}

template <bool ignoreNullKeys>
int32_t HashTable<ignoreNullKeys>::listProbedRows(
    RowsIterator* iter,
    int32_t maxRows,
    uint64_t maxBytes,
    char** rows) {
  return listRows<RowContainer::ProbeType::kProbed>(
      iter, maxRows, maxBytes, rows);
}

template <bool ignoreNullKeys>
void HashTable<ignoreNullKeys>::erase(folly::Range<char**> rows) {
  auto numRows = rows.size();
  raw_vector<uint64_t> hashes;
  hashes.resize(numRows);

  for (int32_t i = 0; i < hashers_.size(); ++i) {
    auto& hasher = hashers_[i];
    if (hashMode_ == HashMode::kHash) {
      rows_->hash(i, rows, i > 0, hashes.data());
    } else {
      auto column = rows_->columnAt(i);
      if (!hasher->computeValueIdsForRows(
              rows.data(),
              numRows,
              column.offset(),
              column.nullByte(),
              ignoreNullKeys ? 0 : column.nullMask(),
              hashes)) {
        VELOX_FAIL("Value ids in erase must exist for all keys");
      }
    }
  }
  eraseWithHashes(rows, hashes.data());
}

template <bool ignoreNullKeys>
void HashTable<ignoreNullKeys>::eraseWithHashes(
    folly::Range<char**> rows,
    uint64_t* hashes) {
  auto numRows = rows.size();
  if (hashMode_ == HashMode::kArray) {
    for (auto i = 0; i < numRows; ++i) {
      DCHECK(hashes[i] < size_);
      table_[hashes[i]] = nullptr;
    }
  } else {
    if (hashMode_ == HashMode::kNormalizedKey) {
      for (auto i = 0; i < numRows; ++i) {
        hashes[i] = mixNormalizedKey(hashes[i], sizeBits_);
      }
    }

    ProbeState state;
    for (auto i = 0; i < numRows; ++i) {
      state.preProbe(*this, hashes[i], i);

      state.firstProbe<ProbeState::Operation::kErase>(*this, 0);
      state.fullProbe<ProbeState::Operation::kErase>(
          *this,
          0,
          [&](const char* group, int32_t row) { return rows[row] == group; },
          [&](int32_t /*index*/, int32_t /*row*/) { return nullptr; },
          false);
    }
  }
  numDistinct_ -= numRows;
  rows_->eraseRows(rows);
}

template class HashTable<true>;
template class HashTable<false>;

} // namespace facebook::velox::exec
