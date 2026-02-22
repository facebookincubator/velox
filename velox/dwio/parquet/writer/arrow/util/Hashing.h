/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

// Adapted from Apache Arrow.

// Private header, not to be exported.

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "arrow/array/builder_binary.h"
#include "arrow/buffer_builder.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/type_fwd.h"
#include "arrow/type_traits.h"
#include "arrow/util/bit_util.h"
#include "arrow/util/bitmap_builders.h"
#include "arrow/util/endian.h"
#include "arrow/util/macros.h"
#include "arrow/util/ubsan.h"

#include "velox/common/base/Exceptions.h"

#define XXH_INLINE_ALL

#include <xxhash.h>

namespace facebook::velox::parquet::arrow::internal {

using ::arrow::BinaryBuilder;
using ::arrow::enable_if_t;
using ::arrow::MemoryPool;
using ::arrow::Status;

// XXX would it help to have a 32-bit hash value on large datasets?
typedef uint64_t hash_t;

// Notes about the choice of a hash function.
// - XXH3 is extremely fast on most data sizes, from small to huge;
//   Faster even than HW CRC-based hashing schemes.
// - Our custom hash function for tiny values (< 16 bytes) is still.
//   Significantly faster (~30%), at least on this machine and compiler.

template <uint64_t AlgNum>
inline hash_t computeStringHash(const void* data, int64_t length);

/// \brief A hash function for bitmaps that can handle offsets and lengths in
/// terms of number of bits. The hash only depends on the bits actually hashed.
///
/// It's the caller's responsibility to ensure that bits_offset + num_bits are
/// readable from the bitmap.
///
/// \pre bits_offset >= 0.
/// \pre num_bits >= 0.
/// \pre (bits_offset + num_bits + 7) / 8 <= readable length in bytes from
/// bitmap.
///
/// \param bitmap The pointer to the bitmap.
/// \param seed The seed for the hash function (useful when chaining hash.
/// Functions). \param bits_offset The offset in bits relative to the start of.
/// The bitmap. \param num_bits The number of bits after the offset to be.
/// Hashed.
ARROW_EXPORT hash_t computeBitmapHash(
    const uint8_t* bitmap,
    hash_t seed,
    int64_t bitsOffset,
    int64_t numBits);

template <typename Scalar, uint64_t AlgNum>
struct ScalarHelperBase {
  static bool compareScalars(Scalar u, Scalar v) {
    return u == v;
  }

  static hash_t computeHash(const Scalar& value) {
    // Generic hash computation for scalars.  Simply apply the string hash.
    // To the bit representation of the value.

    // XXX in the case of FP values, we'd like equal values to have the same
    // hash, even if they have different bit representations...
    return computeStringHash<AlgNum>(&value, sizeof(value));
  }
};

template <typename Scalar, uint64_t AlgNum = 0, typename Enable = void>
struct ScalarHelper : public ScalarHelperBase<Scalar, AlgNum> {};

template <typename Scalar, uint64_t AlgNum>
struct ScalarHelper<
    Scalar,
    AlgNum,
    enable_if_t<std::is_integral<Scalar>::value>>
    : public ScalarHelperBase<Scalar, AlgNum> {
  // ScalarHelper specialization for integers.

  static hash_t computeHash(const Scalar& value) {
    // Faster hash computation for integers.

    // Two of xxhash's prime multipliers (which are chosen for their.
    // bit dispersion properties)
    static constexpr uint64_t multipliers[] = {
        11400714785074694791ULL, 14029467366897019727ULL};

    // Multiplying by the prime number mixes the low bits into the high bits,.
    // Then byte-swapping (which is a single CPU instruction) allows the.
    // Combined high and low bits to participate in the initial hash table.
    // Index.
    auto h = static_cast<hash_t>(value);
    return ::arrow::bit_util::ByteSwap(multipliers[AlgNum] * h);
  }
};

template <typename Scalar, uint64_t AlgNum>
struct ScalarHelper<
    Scalar,
    AlgNum,
    enable_if_t<std::is_same<std::string_view, Scalar>::value>>
    : public ScalarHelperBase<Scalar, AlgNum> {
  // ScalarHelper specialization for std::string_view.

  static hash_t computeHash(std::string_view value) {
    return computeStringHash<AlgNum>(
        value.data(), static_cast<int64_t>(value.size()));
  }
};

template <typename Scalar, uint64_t AlgNum>
struct ScalarHelper<
    Scalar,
    AlgNum,
    enable_if_t<std::is_floating_point<Scalar>::value>>
    : public ScalarHelperBase<Scalar, AlgNum> {
  // ScalarHelper specialization for reals.

  static bool compareScalars(Scalar u, Scalar v) {
    if (std::isnan(u)) {
      // XXX should we do a bit-precise comparison?
      return std::isnan(v);
    }
    return u == v;
  }
};

template <uint64_t AlgNum = 0>
hash_t computeStringHash(const void* data, int64_t length) {
  if (ARROW_PREDICT_TRUE(length <= 16)) {
    // Specialize for small hash strings, as they are quite common as.
    // Hash table keys.  Even XXH3 isn't quite as fast.
    auto p = reinterpret_cast<const uint8_t*>(data);
    auto n = static_cast<uint32_t>(length);
    if (n <= 8) {
      if (n <= 3) {
        if (n == 0) {
          return 1U;
        }
        uint32_t x = (n << 24) ^ (p[0] << 16) ^ (p[n / 2] << 8) ^ p[n - 1];
        return ScalarHelper<uint32_t, AlgNum>::computeHash(x);
      }
      // 4 <= Length <= 8.
      // We can read the string as two overlapping 32-bit ints, apply.
      // Different hash functions to each of them in parallel, then XOR.
      // The results.
      uint32_t x, y;
      hash_t hx, hy;
      x = ::arrow::util::SafeLoadAs<uint32_t>(p + n - 4);
      y = ::arrow::util::SafeLoadAs<uint32_t>(p);
      hx = ScalarHelper<uint32_t, AlgNum>::computeHash(x);
      hy = ScalarHelper<uint32_t, AlgNum ^ 1>::computeHash(y);
      return n ^ hx ^ hy;
    }
    // 8 <= Length <= 16.
    // Apply the same principle as above.
    uint64_t x, y;
    hash_t hx, hy;
    x = ::arrow::util::SafeLoadAs<uint64_t>(p + n - 8);
    y = ::arrow::util::SafeLoadAs<uint64_t>(p);
    hx = ScalarHelper<uint64_t, AlgNum>::computeHash(x);
    hy = ScalarHelper<uint64_t, AlgNum ^ 1>::computeHash(y);
    return n ^ hx ^ hy;
  }

#if XXH3_SECRET_SIZE_MIN != 136
#error XXH3_SECRET_SIZE_MIN changed, please fix kXxh3Secrets
#endif

  // XXH3_64bits_withSeed generates a secret based on the seed, which is too.
  // Slow. Instead, we use hard-coded random secrets.  To maximize cache.
  // Efficiency, they reuse the same memory area.
  static constexpr unsigned char kXxh3Secrets[XXH3_SECRET_SIZE_MIN + 1] = {
      0xe7, 0x8b, 0x13, 0xf9, 0xfc, 0xb5, 0x8e, 0xef, 0x81, 0x48, 0x2c, 0xbf,
      0xf9, 0x9f, 0xc1, 0x1e, 0x43, 0x6d, 0xbf, 0xa6, 0x6d, 0xb5, 0x72, 0xbc,
      0x97, 0xd8, 0x61, 0x24, 0x0f, 0x12, 0xe3, 0x05, 0x21, 0xf7, 0x5c, 0x66,
      0x67, 0xa5, 0x65, 0x03, 0x96, 0x26, 0x69, 0xd8, 0x29, 0x20, 0xf8, 0xc7,
      0xb0, 0x3d, 0xdd, 0x7d, 0x18, 0xa0, 0x60, 0x75, 0x92, 0xa4, 0xce, 0xba,
      0xc0, 0x77, 0xf4, 0xac, 0xb7, 0x03, 0x53, 0xf0, 0x98, 0xce, 0xe6, 0x2b,
      0x20, 0xc7, 0x82, 0x91, 0xab, 0xbf, 0x68, 0x5c, 0x62, 0x4d, 0x33, 0xa3,
      0xe1, 0xb3, 0xff, 0x97, 0x54, 0x4c, 0x44, 0x34, 0xb5, 0xb9, 0x32, 0x4c,
      0x75, 0x42, 0x89, 0x53, 0x94, 0xd4, 0x9f, 0x2b, 0x76, 0x4d, 0x4e, 0xe6,
      0xfa, 0x15, 0x3e, 0xc1, 0xdb, 0x71, 0x4b, 0x2c, 0x94, 0xf5, 0xfc, 0x8c,
      0x89, 0x4b, 0xfb, 0xc1, 0x82, 0xa5, 0x6a, 0x53, 0xf9, 0x4a, 0xba, 0xce,
      0x1f, 0xc0, 0x97, 0x1a, 0x87};

  static_assert(AlgNum < 2, "AlgNum too large");
  static constexpr auto secret = kXxh3Secrets + AlgNum;
  return XXH3_64bits_withSecret(
      data, static_cast<size_t>(length), secret, XXH3_SECRET_SIZE_MIN);
}

// XXX add a HashEq<ArrowType> struct with both hash and compare functions?

// ----------------------------------------------------------------------.
// An open-addressing insert-only hash table (no deletes)

template <typename Payload>
class HashTable {
 public:
  static constexpr hash_t kSentinel = 0ULL;
  static constexpr int64_t kLoadFactor = 2UL;

  struct Entry {
    hash_t h;
    Payload payload;

    // An entry is valid if the hash is different from the sentinel value.
    operator bool() const {
      return h != kSentinel;
    }
  };

  HashTable(MemoryPool* pool, uint64_t capacity) : entriesBuilder_(pool) {
    VELOX_DCHECK_NOT_NULL(pool);
    // Minimum of 32 elements.
    capacity = std::max<uint64_t>(capacity, 32UL);
    capacity_ = ::arrow::bit_util::NextPower2(capacity);
    capacityMask_ = capacity_ - 1;
    size_ = 0;

    auto status = upsizeBuffer(capacity_);
    VELOX_DCHECK(status.ok(), status.ToString());
  }

  // Lookup with non-linear probing.
  // Cmp_func should have signature bool(const Payload*).
  // Return a (Entry*, found) pair.
  template <typename CmpFunc>
  std::pair<Entry*, bool> lookup(hash_t h, CmpFunc&& cmpFunc) {
    auto p = lookup<DoCompare, CmpFunc>(
        h, entries_, capacityMask_, std::forward<CmpFunc>(cmpFunc));
    return {&entries_[p.first], p.second};
  }

  template <typename CmpFunc>
  std::pair<const Entry*, bool> lookup(hash_t h, CmpFunc&& cmpFunc) const {
    auto p = lookup<DoCompare, CmpFunc>(
        h, entries_, capacityMask_, std::forward<CmpFunc>(cmpFunc));
    return {&entries_[p.first], p.second};
  }

  Status insert(Entry* entry, hash_t h, const Payload& payload) {
    // Ensure entry is empty before inserting.
    assert(!*entry);
    entry->h = fixHash(h);
    entry->payload = payload;
    ++size_;

    if (ARROW_PREDICT_FALSE(needUpsizing())) {
      // Resize less frequently since it is expensive.
      return upsize(capacity_ * kLoadFactor * 2);
    }
    return Status::OK();
  }

  uint64_t size() const {
    return size_;
  }

  // Visit all non-empty entries in the table.
  // The visit_func should have signature void(const Entry*)
  template <typename VisitFunc>
  void visitEntries(VisitFunc&& visitFunc) const {
    for (uint64_t i = 0; i < capacity_; i++) {
      const auto& entry = entries_[i];
      if (entry) {
        visitFunc(&entry);
      }
    }
  }

 protected:
  // NoCompare is for when the value is known not to exist in the table.
  enum CompareKind { DoCompare, NoCompare };

  // The workhorse lookup function.
  template <CompareKind CKind, typename CmpFunc>
  std::pair<uint64_t, bool> lookup(
      hash_t h,
      const Entry* entries,
      uint64_t sizeMask,
      CmpFunc&& cmpFunc) const {
    static constexpr uint8_t perturbShift = 5;

    uint64_t index, perturb;
    const Entry* entry;

    h = fixHash(h);
    index = h & sizeMask;
    perturb = (h >> perturbShift) + 1U;

    while (true) {
      entry = &entries[index];
      if (compareEntry<CKind, CmpFunc>(
              h, entry, std::forward<CmpFunc>(cmpFunc))) {
        // Found.
        return {index, true};
      }
      if (entry->h == kSentinel) {
        // Empty slot.
        return {index, false};
      }

      // Perturbation logic inspired from CPython's set / dict object.
      // The goal is that all 64 bits of the unmasked hash value eventually.
      // Participate in the probing sequence, to minimize clustering.
      index = (index + perturb) & sizeMask;
      perturb = (perturb >> perturbShift) + 1U;
    }
  }

  template <CompareKind CKind, typename CmpFunc>
  bool compareEntry(hash_t h, const Entry* entry, CmpFunc&& cmpFunc) const {
    if (CKind == NoCompare) {
      return false;
    } else {
      return entry->h == h && cmpFunc(&entry->payload);
    }
  }

  bool needUpsizing() const {
    // Keep the load factor <= 1/2.
    return size_ * kLoadFactor >= capacity_;
  }

  Status upsizeBuffer(uint64_t capacity) {
    RETURN_NOT_OK(entriesBuilder_.Resize(capacity));
    entries_ = entriesBuilder_.mutable_data();
    memset(static_cast<void*>(entries_), 0, capacity * sizeof(Entry));

    return Status::OK();
  }

  Status upsize(uint64_t newCapacity) {
    assert(newCapacity > capacity_);
    uint64_t newMask = newCapacity - 1;
    assert((newCapacity & newMask) == 0); // it's a power of two

    // Stash old entries and seal builder, effectively resetting the Buffer.
    const Entry* oldEntries = entries_;
    ARROW_ASSIGN_OR_RAISE(
        auto previous, entriesBuilder_.FinishWithLength(capacity_));
    // Allocate new buffer.
    RETURN_NOT_OK(upsizeBuffer(newCapacity));

    for (uint64_t i = 0; i < capacity_; i++) {
      const auto& entry = oldEntries[i];
      if (entry) {
        // Dummy compare function will not be called.
        auto p = lookup<NoCompare>(
            entry.h, entries_, newMask, [](const Payload*) { return false; });
        // Lookup<NoCompare> (and CompareEntry<NoCompare>) ensure that an.
        // Empty slots is always returned.
        assert(!p.second);
        entries_[p.first] = entry;
      }
    }
    capacity_ = newCapacity;
    capacityMask_ = newMask;

    return Status::OK();
  }

  hash_t fixHash(hash_t h) const {
    return (h == kSentinel) ? 42U : h;
  }

  // The number of slots available in the hash table array.
  uint64_t capacity_;
  uint64_t capacityMask_;
  // The number of used slots in the hash table array.
  uint64_t size_;

  Entry* entries_;
  ::arrow::TypedBufferBuilder<Entry> entriesBuilder_;
};

// XXX typedef memo_index_t int32_t ?

constexpr int32_t kKeyNotFound = -1;

// ----------------------------------------------------------------------.
// A base class for memoization table.

class MemoTable {
 public:
  virtual ~MemoTable() = default;

  virtual int32_t size() const = 0;
};

// ----------------------------------------------------------------------.
// A memoization table for memory-cheap scalar values.

// The memoization table remembers and allows to look up the insertion.
// Index for each key.

template <
    typename Scalar,
    template <class> class HashTableTemplateType = HashTable>
class ScalarMemoTable : public MemoTable {
 public:
  explicit ScalarMemoTable(MemoryPool* pool, int64_t entries = 0)
      : hashTable_(pool, static_cast<uint64_t>(entries)) {}

  int32_t get(const Scalar& value) const {
    auto cmpFunc = [value](const Payload* payload) -> bool {
      return ScalarHelper<Scalar, 0>::compareScalars(payload->value, value);
    };
    hash_t h = computeHash(value);
    auto p = hashTable_.lookup(h, cmpFunc);
    if (p.second) {
      return p.first->payload.memoIndex;
    } else {
      return kKeyNotFound;
    }
  }

  template <typename Func1, typename Func2>
  Status getOrInsert(
      const Scalar& value,
      Func1&& onFound,
      Func2&& onNotFound,
      int32_t* outMemoIndex) {
    auto cmpFunc = [value](const Payload* payload) -> bool {
      return ScalarHelper<Scalar, 0>::compareScalars(value, payload->value);
    };
    hash_t h = computeHash(value);
    auto p = hashTable_.lookup(h, cmpFunc);
    int32_t memoIndex;
    if (p.second) {
      memoIndex = p.first->payload.memoIndex;
      onFound(memoIndex);
    } else {
      memoIndex = size();
      RETURN_NOT_OK(hashTable_.insert(p.first, h, {value, memoIndex}));
      onNotFound(memoIndex);
    }
    *outMemoIndex = memoIndex;
    return Status::OK();
  }

  Status getOrInsert(const Scalar& value, int32_t* outMemoIndex) {
    return getOrInsert(value, [](int32_t i) {}, [](int32_t i) {}, outMemoIndex);
  }

  int32_t getNull() const {
    return nullIndex_;
  }

  template <typename Func1, typename Func2>
  int32_t getOrInsertNull(Func1&& onFound, Func2&& onNotFound) {
    int32_t memoIndex = getNull();
    if (memoIndex != kKeyNotFound) {
      onFound(memoIndex);
    } else {
      nullIndex_ = memoIndex = size();
      onNotFound(memoIndex);
    }
    return memoIndex;
  }

  int32_t getOrInsertNull() {
    return getOrInsertNull([](int32_t i) {}, [](int32_t i) {});
  }

  // The number of entries in the memo table +1 if null was added.
  // (which is also 1 + the largest memo index)
  int32_t size() const override {
    return static_cast<int32_t>(hashTable_.size()) +
        (getNull() != kKeyNotFound);
  }

  // Copy values starting from index `start` into `out_data`.
  void copyValues(int32_t start, Scalar* outData) const {
    hashTable_.visitEntries([=](const HashTableEntry* entry) {
      int32_t index = entry->payload.memoIndex - start;
      if (index >= 0) {
        outData[index] = entry->payload.value;
      }
    });
    // Zero-initialize the null entry.
    if (nullIndex_ != kKeyNotFound) {
      int32_t index = nullIndex_ - start;
      if (index >= 0) {
        outData[index] = Scalar{};
      }
    }
  }

  void copyValues(Scalar* outData) const {
    copyValues(0, outData);
  }

 protected:
  struct Payload {
    Scalar value;
    int32_t memoIndex;
  };

  using HashTableType = HashTableTemplateType<Payload>;
  using HashTableEntry = typename HashTableType::Entry;
  HashTableType hashTable_;
  int32_t nullIndex_ = kKeyNotFound;

  hash_t computeHash(const Scalar& value) const {
    return ScalarHelper<Scalar, 0>::computeHash(value);
  }

 public:
  // Defined here so that `HashTableType` is visible.
  // Merge entries from `other_table` into `this->hash_table_`.
  Status mergeTable(const ScalarMemoTable& otherTable) {
    const HashTableType& otherHashtable = otherTable.hashTable_;

    otherHashtable.visitEntries([this](const HashTableEntry* otherEntry) {
      int32_t unused;
      auto status = this->getOrInsert(otherEntry->payload.value, &unused);
      VELOX_DCHECK(status.ok(), status.ToString());
    });
    // TODO: ARROW-17074 - implement proper error handling.
    return Status::OK();
  }
};

// ----------------------------------------------------------------------.
// A memoization table for small scalar values, using direct indexing.

template <typename Scalar, typename Enable = void>
struct SmallScalarTraits {};

template <>
struct SmallScalarTraits<bool> {
  static constexpr int32_t cardinality = 2;

  static uint32_t asIndex(bool value) {
    return value ? 1 : 0;
  }
};

template <typename Scalar>
struct SmallScalarTraits<Scalar, enable_if_t<std::is_integral<Scalar>::value>> {
  using Unsigned = typename std::make_unsigned<Scalar>::type;

  static constexpr int32_t cardinality =
      1U + std::numeric_limits<Unsigned>::max();

  static uint32_t asIndex(Scalar value) {
    return static_cast<Unsigned>(value);
  }
};

template <
    typename Scalar,
    template <class> class HashTableTemplateType = HashTable>
class SmallScalarMemoTable : public MemoTable {
 public:
  explicit SmallScalarMemoTable(MemoryPool* pool, int64_t entries = 0) {
    std::fill(valueToIndex_, valueToIndex_ + cardinality + 1, kKeyNotFound);
    indexToValue_.reserve(cardinality);
  }

  int32_t get(const Scalar value) const {
    auto valueIndex = asIndex(value);
    return valueToIndex_[valueIndex];
  }

  template <typename Func1, typename Func2>
  Status getOrInsert(
      const Scalar value,
      Func1&& onFound,
      Func2&& onNotFound,
      int32_t* outMemoIndex) {
    auto valueIndex = asIndex(value);
    auto memoIndex = valueToIndex_[valueIndex];
    if (memoIndex == kKeyNotFound) {
      memoIndex = static_cast<int32_t>(indexToValue_.size());
      indexToValue_.push_back(value);
      valueToIndex_[valueIndex] = memoIndex;
      VELOX_DCHECK_LT(memoIndex, cardinality + 1);
      onNotFound(memoIndex);
    } else {
      onFound(memoIndex);
    }
    *outMemoIndex = memoIndex;
    return Status::OK();
  }

  Status getOrInsert(const Scalar value, int32_t* outMemoIndex) {
    return getOrInsert(value, [](int32_t i) {}, [](int32_t i) {}, outMemoIndex);
  }

  int32_t getNull() const {
    return valueToIndex_[cardinality];
  }

  template <typename Func1, typename Func2>
  int32_t getOrInsertNull(Func1&& onFound, Func2&& onNotFound) {
    auto memoIndex = getNull();
    if (memoIndex == kKeyNotFound) {
      memoIndex = valueToIndex_[cardinality] = size();
      indexToValue_.push_back(0);
      onNotFound(memoIndex);
    } else {
      onFound(memoIndex);
    }
    return memoIndex;
  }

  int32_t getOrInsertNull() {
    return getOrInsertNull([](int32_t i) {}, [](int32_t i) {});
  }

  // The number of entries in the memo table.
  // (which is also 1 + the largest memo index)
  int32_t size() const override {
    return static_cast<int32_t>(indexToValue_.size());
  }

  // Merge entries from `other_table` into `this`.
  Status mergeTable(const SmallScalarMemoTable& otherTable) {
    for (const Scalar& otherVal : otherTable.indexToValue_) {
      int32_t unused;
      RETURN_NOT_OK(this->getOrInsert(otherVal, &unused));
    }
    return Status::OK();
  }

  // Copy values starting from index `start` into `out_data`.
  void copyValues(int32_t start, Scalar* outData) const {
    VELOX_DCHECK_GE(start, 0);
    VELOX_DCHECK_LE(static_cast<size_t>(start), indexToValue_.size());
    int64_t offset = start * static_cast<int32_t>(sizeof(Scalar));
    memcpy(
        outData,
        indexToValue_.data() + offset,
        (size() - start) * sizeof(Scalar));
  }

  void copyValues(Scalar* outData) const {
    copyValues(0, outData);
  }

  const std::vector<Scalar>& values() const {
    return indexToValue_;
  }

 protected:
  static constexpr auto cardinality = SmallScalarTraits<Scalar>::cardinality;
  static_assert(
      cardinality <= 256,
      "cardinality too large for direct-addressed table");

  uint32_t asIndex(Scalar value) const {
    return SmallScalarTraits<Scalar>::asIndex(value);
  }

  // The last index is reserved for the null element.
  int32_t valueToIndex_[cardinality + 1];
  std::vector<Scalar> indexToValue_;
};

// ----------------------------------------------------------------------.
// A memoization table for variable-sized binary data.

template <typename BinaryBuilderT>
class BinaryMemoTable : public MemoTable {
 public:
  using BuilderOffsetType = typename BinaryBuilderT::offset_type;
  explicit BinaryMemoTable(
      MemoryPool* pool,
      int64_t entries = 0,
      int64_t valuesSize = -1)
      : hashTable_(pool, static_cast<uint64_t>(entries)), binaryBuilder_(pool) {
    const int64_t dataSize = (valuesSize < 0) ? entries * 4 : valuesSize;
    auto status = binaryBuilder_.Reserve(entries);
    VELOX_DCHECK(status.ok(), status.ToString());
    status = binaryBuilder_.ReserveData(dataSize);
    VELOX_DCHECK(status.ok(), status.ToString());
  }

  int32_t get(const void* data, BuilderOffsetType length) const {
    hash_t h = computeStringHash<0>(data, length);
    auto p = lookup(h, data, length);
    if (p.second) {
      return p.first->payload.memoIndex;
    } else {
      return kKeyNotFound;
    }
  }

  int32_t get(std::string_view value) const {
    return get(value.data(), static_cast<BuilderOffsetType>(value.length()));
  }

  template <typename Func1, typename Func2>
  Status getOrInsert(
      const void* data,
      BuilderOffsetType length,
      Func1&& onFound,
      Func2&& onNotFound,
      int32_t* outMemoIndex) {
    hash_t h = computeStringHash<0>(data, length);
    auto p = lookup(h, data, length);
    int32_t memoIndex;
    if (p.second) {
      memoIndex = p.first->payload.memoIndex;
      onFound(memoIndex);
    } else {
      memoIndex = size();
      // Insert string value.
      RETURN_NOT_OK(
          binaryBuilder_.Append(static_cast<const char*>(data), length));
      // Insert hash entry.
      RETURN_NOT_OK(hashTable_.insert(
          const_cast<HashTableEntry*>(p.first), h, {memoIndex}));

      onNotFound(memoIndex);
    }
    *outMemoIndex = memoIndex;
    return Status::OK();
  }

  template <typename Func1, typename Func2>
  Status getOrInsert(
      std::string_view value,
      Func1&& onFound,
      Func2&& onNotFound,
      int32_t* outMemoIndex) {
    return getOrInsert(
        value.data(),
        static_cast<BuilderOffsetType>(value.length()),
        std::forward<Func1>(onFound),
        std::forward<Func2>(onNotFound),
        outMemoIndex);
  }

  Status getOrInsert(
      const void* data,
      BuilderOffsetType length,
      int32_t* outMemoIndex) {
    return getOrInsert(
        data, length, [](int32_t i) {}, [](int32_t i) {}, outMemoIndex);
  }

  Status getOrInsert(std::string_view value, int32_t* outMemoIndex) {
    return getOrInsert(
        value.data(),
        static_cast<BuilderOffsetType>(value.length()),
        outMemoIndex);
  }

  int32_t getNull() const {
    return nullIndex_;
  }

  template <typename Func1, typename Func2>
  int32_t getOrInsertNull(Func1&& onFound, Func2&& onNotFound) {
    int32_t memoIndex = getNull();
    if (memoIndex == kKeyNotFound) {
      memoIndex = nullIndex_ = size();
      auto status = binaryBuilder_.AppendNull();
      VELOX_DCHECK(status.ok(), status.ToString());
      onNotFound(memoIndex);
    } else {
      onFound(memoIndex);
    }
    return memoIndex;
  }

  int32_t getOrInsertNull() {
    return getOrInsertNull([](int32_t i) {}, [](int32_t i) {});
  }

  // The number of entries in the memo table.
  // (which is also 1 + the largest memo index)
  int32_t size() const override {
    return static_cast<int32_t>(
        hashTable_.size() + (getNull() != kKeyNotFound));
  }

  int64_t valuesSize() const {
    return binaryBuilder_.value_data_length();
  }

  // Copy (n + 1) offsets starting from index `start` into `out_data`.
  template <class Offset>
  void copyOffsets(int32_t start, Offset* outData) const {
    VELOX_DCHECK_LE(start, size());

    const BuilderOffsetType* offsets = binaryBuilder_.offsets_data();
    const BuilderOffsetType delta =
        start < binaryBuilder_.length() ? offsets[start] : 0;
    for (int32_t i = start; i < size(); ++i) {
      const BuilderOffsetType adjustedOffset = offsets[i] - delta;
      Offset castOffset = static_cast<Offset>(adjustedOffset);
      assert(
          static_cast<BuilderOffsetType>(castOffset) ==
          adjustedOffset); // avoid truncation
      *outData++ = castOffset;
    }

    // Copy last value since BinaryBuilder only materializes it on in Finish()
    *outData = static_cast<Offset>(binaryBuilder_.value_data_length() - delta);
  }

  template <class Offset>
  void copyOffsets(Offset* outData) const {
    copyOffsets(0, outData);
  }

  // Copy values starting from index `start` into `out_data`.
  void copyValues(int32_t start, uint8_t* outData) const {
    copyValues(start, -1, outData);
  }

  // Same as above, but check output size in debug mode.
  void copyValues(int32_t start, int64_t outSize, uint8_t* outData) const {
    VELOX_DCHECK_LE(start, size());

    // The absolute byte offset of `start` value in the binary buffer.
    const BuilderOffsetType offset = binaryBuilder_.offsets_data()[start];
    const auto length =
        binaryBuilder_.value_data_length() - static_cast<size_t>(offset);

    if (outSize != -1) {
      assert(static_cast<int64_t>(length) <= outSize);
    }

    auto view = binaryBuilder_.GetView(start);
    memcpy(outData, view.data(), length);
  }

  void copyValues(uint8_t* outData) const {
    copyValues(0, -1, outData);
  }

  void copyValues(int64_t outSize, uint8_t* outData) const {
    copyValues(0, outSize, outData);
  }

  void copyFixedWidthValues(
      int32_t start,
      int32_t widthSize,
      int64_t outSize,
      uint8_t* outData) const {
    // This method exists to cope with the fact that the BinaryMemoTable does.
    // Not know the fixed width when inserting the null value. The data.
    // Buffer hold a zero length string for the null value (if found).
    //
    // Thus, the method will properly inject an empty value of the proper width.
    // In the output buffer.
    //
    if (start >= size()) {
      return;
    }

    int32_t nullIndex = getNull();
    if (nullIndex < start) {
      // Nothing to skip, proceed as usual.
      copyValues(start, outSize, outData);
      return;
    }

    BuilderOffsetType leftOffset = binaryBuilder_.offsets_data()[start];

    // Ensure that the data length is exactly missing width_size bytes to fit.
    // In the expected output (n_values * width_size).
#ifndef NDEBUG
    int64_t dataLength = valuesSize() - static_cast<size_t>(leftOffset);
    assert(dataLength + widthSize == outSize);
    ARROW_UNUSED(dataLength);
#endif

    auto inData = binaryBuilder_.value_data() + leftOffset;
    // The null use 0-length in the data, slice the data in 2 and skip by.
    // Width_size in out_data. [part_1][width_size][part_2].
    auto nullDataOffset = binaryBuilder_.offsets_data()[nullIndex];
    auto leftSize = nullDataOffset - leftOffset;
    if (leftSize > 0) {
      memcpy(outData, inData + leftOffset, leftSize);
    }
    // Zero-initialize the null entry.
    memset(outData + leftSize, 0, widthSize);

    auto rightSize = valuesSize() - static_cast<size_t>(nullDataOffset);
    if (rightSize > 0) {
      // Skip the null fixed size value.
      auto outOffset = leftSize + widthSize;
      assert(outData + outOffset + rightSize == outData + outSize);
      memcpy(outData + outOffset, inData + nullDataOffset, rightSize);
    }
  }

  // Visit the stored values in insertion order.
  // The visitor function should have the signature `void(std::string_view)`.
  // Or `void(const std::string_view&)`.
  template <typename VisitFunc>
  void visitValues(int32_t start, VisitFunc&& visit) const {
    for (int32_t i = start; i < size(); ++i) {
      auto sv = binaryBuilder_.GetView(i);
      visit(std::string_view(sv.data(), sv.size()));
    }
  }

 protected:
  struct Payload {
    int32_t memoIndex;
  };

  using HashTableType = HashTable<Payload>;
  using HashTableEntry = typename HashTable<Payload>::Entry;
  HashTableType hashTable_;
  BinaryBuilderT binaryBuilder_;

  int32_t nullIndex_ = kKeyNotFound;

  std::pair<const HashTableEntry*, bool>
  lookup(hash_t h, const void* data, BuilderOffsetType length) const {
    auto cmpFunc = [&](const Payload* payload) {
      auto lhs = binaryBuilder_.GetView(payload->memoIndex);
      auto rhs = std::string_view(static_cast<const char*>(data), length);
      return lhs == rhs;
    };
    return hashTable_.lookup(h, cmpFunc);
  }

 public:
  Status mergeTable(const BinaryMemoTable& otherTable) {
    otherTable.visitValues(0, [this](std::string_view otherValue) {
      int32_t unused;
      auto status = this->getOrInsert(otherValue, &unused);
      VELOX_DCHECK(status.ok(), status.ToString());
    });
    return Status::OK();
  }
};

template <typename T, typename Enable = void>
struct HashTraits {};

template <>
struct HashTraits<::arrow::BooleanType> {
  using MemoTableType = SmallScalarMemoTable<bool>;
};

template <typename T>
struct HashTraits<T, ::arrow::enable_if_8bit_int<T>> {
  using CType = typename T::CType;
  using MemoTableType = SmallScalarMemoTable<typename T::CType>;
};

template <typename T>
struct HashTraits<
    T,
    enable_if_t<
        ::arrow::has_c_type<T>::value && !::arrow::is_8bit_int<T>::value>> {
  using CType = typename T::CType;
  using MemoTableType = ScalarMemoTable<CType, HashTable>;
};

template <typename T>
struct HashTraits<
    T,
    enable_if_t<
        ::arrow::has_string_view<T>::value &&
        !std::is_base_of<::arrow::LargeBinaryType, T>::value>> {
  using MemoTableType = BinaryMemoTable<BinaryBuilder>;
};

template <typename T>
struct HashTraits<T, ::arrow::enable_if_decimal<T>> {
  using MemoTableType = BinaryMemoTable<BinaryBuilder>;
};

template <typename T>
struct HashTraits<
    T,
    enable_if_t<std::is_base_of<::arrow::LargeBinaryType, T>::value>> {
  using MemoTableType = BinaryMemoTable<::arrow::LargeBinaryBuilder>;
};

template <typename MemoTableType>
static inline Status computeNullBitmap(
    MemoryPool* pool,
    const MemoTableType& memoTable,
    int64_t startOffset,
    int64_t* nullCount,
    std::shared_ptr<::arrow::Buffer>* nullBitmap) {
  int64_t dictLength = static_cast<int64_t>(memoTable.size()) - startOffset;
  int64_t nullIndex = memoTable.getNull();

  *nullCount = 0;
  *nullBitmap = nullptr;

  if (nullIndex != kKeyNotFound && nullIndex >= startOffset) {
    nullIndex -= startOffset;
    *nullCount = 1;
    ARROW_ASSIGN_OR_RAISE(
        *nullBitmap,
        ::arrow::internal::BitmapAllButOne(pool, dictLength, nullIndex));
  }

  return Status::OK();
}

struct StringViewHash {
  // Std::hash compatible hasher for use with std::unordered_*.
  // (The std::hash specialization provided by nonstd constructs std::string.
  // temporaries then invokes std::hash<std::string> against those)
  hash_t operator()(std::string_view value) const {
    return computeStringHash<0>(
        value.data(), static_cast<int64_t>(value.size()));
  }
};

} // namespace facebook::velox::parquet::arrow::internal
