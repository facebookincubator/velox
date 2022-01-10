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

#include "velox/exec/VectorHasher.h"
#include "velox/common/base/BitUtil.h"
#include "velox/common/base/Portability.h"
#include "velox/common/base/SimdUtil.h"
#include "velox/common/memory/HashStringAllocator.h"

DEFINE_bool(enable_str_simd, true, "Enable StringView SIMD hash");

namespace facebook::velox::exec {

#define VALUE_ID_TYPE_DISPATCH(TEMPLATE_FUNC, typeKind, ...)             \
  [&]() {                                                                \
    switch (typeKind) {                                                  \
      case TypeKind::BOOLEAN: {                                          \
        return TEMPLATE_FUNC<TypeKind::BOOLEAN>(__VA_ARGS__);            \
      }                                                                  \
      case TypeKind::TINYINT: {                                          \
        return TEMPLATE_FUNC<TypeKind::TINYINT>(__VA_ARGS__);            \
      }                                                                  \
      case TypeKind::SMALLINT: {                                         \
        return TEMPLATE_FUNC<TypeKind::SMALLINT>(__VA_ARGS__);           \
      }                                                                  \
      case TypeKind::INTEGER: {                                          \
        return TEMPLATE_FUNC<TypeKind::INTEGER>(__VA_ARGS__);            \
      }                                                                  \
      case TypeKind::BIGINT: {                                           \
        return TEMPLATE_FUNC<TypeKind::BIGINT>(__VA_ARGS__);             \
      }                                                                  \
      case TypeKind::VARCHAR:                                            \
      case TypeKind::VARBINARY: {                                        \
        return TEMPLATE_FUNC<TypeKind::VARCHAR>(__VA_ARGS__);            \
      }                                                                  \
      default:                                                           \
        VELOX_UNREACHABLE(                                               \
            "Unsupported value ID type: ", mapTypeKindToName(typeKind)); \
    }                                                                    \
  }()

using V32 = simd::Vectors<int32_t>;
using V64 = simd::Vectors<int64_t>;

namespace {
template <TypeKind Kind>
uint64_t hashOne(DecodedVector& decoded, vector_size_t index) {
  if (Kind == TypeKind::ROW || Kind == TypeKind::ARRAY ||
      Kind == TypeKind::MAP) {
    // Virtual function call for complex type.
    return decoded.base()->hashValueAt(decoded.index(index));
  }
  // Inlined for scalars.
  using T = typename KindToFlatVector<Kind>::HashRowType;
  return folly::hasher<T>()(decoded.valueAt<T>(index));
}
} // namespace

template <TypeKind Kind>
void VectorHasher::hashValues(
    const SelectivityVector& rows,
    bool mix,
    uint64_t* result) {
  using T = typename TypeTraits<Kind>::NativeType;
  if (decoded_.isConstantMapping()) {
    auto hash = decoded_.isNullAt(rows.begin())
        ? kNullHash
        : hashOne<Kind>(decoded_, rows.begin());
    rows.applyToSelected([&](vector_size_t row) {
      result[row] = mix ? bits::hashMix(result[row], hash) : hash;
    });
  } else if (decoded_.isIdentityMapping()) {
    rows.applyToSelected([&](vector_size_t row) {
      if (decoded_.isNullAt(row)) {
        result[row] = mix ? bits::hashMix(result[row], kNullHash) : kNullHash;
        return;
      }
      auto hash = hashOne<Kind>(decoded_, row);
      result[row] = mix ? bits::hashMix(result[row], hash) : hash;
    });
  } else {
    cachedHashes_.resize(decoded_.base()->size());
    std::fill(cachedHashes_.begin(), cachedHashes_.end(), kNullHash);
    rows.applyToSelected([&](vector_size_t row) {
      if (decoded_.isNullAt(row)) {
        result[row] = mix ? bits::hashMix(result[row], kNullHash) : kNullHash;
        return;
      }
      auto baseIndex = decoded_.index(row);
      uint64_t hash = cachedHashes_[baseIndex];
      if (hash == kNullHash) {
        hash = hashOne<Kind>(decoded_, row);
        cachedHashes_[baseIndex] = hash;
      }
      result[row] = mix ? bits::hashMix(result[row], hash) : hash;
    });
  }
}

template <TypeKind Kind>
bool VectorHasher::makeValueIds(
    const SelectivityVector& rows,
    uint64_t* result) {
  using T = typename TypeTraits<Kind>::NativeType;

  if (decoded_.isConstantMapping()) {
    uint64_t id = decoded_.isNullAt(rows.begin())
        ? 0
        : valueId(decoded_.valueAt<T>(rows.begin()));
    if (id == kUnmappable) {
      analyzeValue(decoded_.valueAt<T>(rows.begin()));
      return false;
    }
    rows.applyToSelected([&](vector_size_t row) INLINE_LAMBDA {
      result[row] = multiplier_ == 1 ? id : result[row] + multiplier_ * id;
    });
    return true;
  }

  if (decoded_.isIdentityMapping()) {
    if (decoded_.mayHaveNulls()) {
      return makeValueIdsFlatWithNulls<T>(rows, result);
    } else {
      return makeValueIdsFlatNoNulls<T>(rows, result);
    }
  }

  if (decoded_.mayHaveNulls()) {
    return makeValueIdsDecoded<T, true>(rows, result);
  } else {
    return makeValueIdsDecoded<T, false>(rows, result);
  }
}

template <>
bool VectorHasher::makeValueIdsFlatNoNulls<bool>(
    const SelectivityVector& rows,
    uint64_t* result) {
  const auto* values = decoded_.data<uint64_t>();
  rows.applyToSelected([&](vector_size_t row) INLINE_LAMBDA {
    bool value = bits::isBitSet(values, row);
    uint64_t id = valueId(value);
    result[row] = multiplier_ == 1 ? id : result[row] + multiplier_ * id;
  });
  return true;
}

template <>
bool VectorHasher::makeValueIdsFlatWithNulls<bool>(
    const SelectivityVector& rows,
    uint64_t* result) {
  const auto* values = decoded_.data<uint64_t>();
  const auto* nulls = decoded_.nulls();
  rows.applyToSelected([&](vector_size_t row) INLINE_LAMBDA {
    if (bits::isBitNull(nulls, row)) {
      if (multiplier_ == 1) {
        result[row] = 0;
      }
      return;
    }
    bool value = bits::isBitSet(values, row);
    uint64_t id = valueId(value);
    result[row] = multiplier_ == 1 ? id : result[row] + multiplier_ * id;
  });
  return true;
}

template <typename T>
bool VectorHasher::makeValueIdsFlatNoNulls(
    const SelectivityVector& rows,
    uint64_t* result) {
  const auto* values = decoded_.values<T>();
  if (isRange_ && tryMapToRange(values, rows, result)) {
    return true;
  }

  bool success = true;
  rows.applyToSelected([&](vector_size_t row) INLINE_LAMBDA {
    T value = values[row];
    if (!success) {
      // If all were not mappable and we do not remove unmappable,
      // we just analyze the remaining so we can decide the hash mode.
      analyzeValue(value);
      return;
    }
    uint64_t id = valueId(value);
    if (id == kUnmappable) {
      success = false;
      analyzeValue(value);
      return;
    }
    result[row] = multiplier_ == 1 ? id : result[row] + multiplier_ * id;
  });

  return success;
}

template <typename T>
bool VectorHasher::makeValueIdsFlatWithNulls(
    const SelectivityVector& rows,
    uint64_t* result) {
  const auto* values = decoded_.values<T>();
  const auto* nulls = decoded_.nulls();

  bool success = true;
  rows.applyToSelected([&](vector_size_t row) INLINE_LAMBDA {
    if (bits::isBitNull(nulls, row)) {
      if (multiplier_ == 1) {
        result[row] = 0;
      }
      return;
    }
    T value = values[row];
    if (!success) {
      // If all were not mappable we just analyze the remaining so we can decide
      // the hash mode.
      analyzeValue(value);
      return;
    }
    uint64_t id = valueId(value);
    if (id == kUnmappable) {
      success = false;
      analyzeValue(value);
      return;
    }
    result[row] = multiplier_ == 1 ? id : result[row] + multiplier_ * id;
  });
  return success;
}

template <typename T, bool mayHaveNulls>
bool VectorHasher::makeValueIdsDecoded(
    const SelectivityVector& rows,
    uint64_t* result) {
  cachedHashes_.resize(decoded_.base()->size());
  std::fill(cachedHashes_.begin(), cachedHashes_.end(), 0);

  auto indices = decoded_.indices();
  auto values = decoded_.values<T>();

  bool success = true;
  rows.applyToSelected([&](vector_size_t row) INLINE_LAMBDA {
    if constexpr (mayHaveNulls) {
      if (decoded_.isNullAt(row)) {
        if (multiplier_ == 1) {
          result[row] = 0;
        }
        return;
      }
    }
    auto baseIndex = indices[row];
    uint64_t id = cachedHashes_[baseIndex];
    if (id == 0) {
      T value = values[baseIndex];

      if (!success) {
        // If all were not mappable we just analyze the remaining so we can
        // decide the hash mode.
        analyzeValue(value);
        return;
      }
      id = valueId(value);
      if (id == kUnmappable) {
        analyzeValue(value);
        success = false;
        return;
      }
      cachedHashes_[baseIndex] = id;
    }
    result[row] = multiplier_ == 1 ? id : result[row] + multiplier_ * id;
  });
  return success;
}

template <>
bool VectorHasher::makeValueIdsDecoded<bool, true>(
    const SelectivityVector& rows,
    uint64_t* result) {
  auto indices = decoded_.indices();
  auto values = decoded_.values<uint64_t>();

  rows.applyToSelected([&](vector_size_t row) INLINE_LAMBDA {
    if (decoded_.isNullAt(row)) {
      if (multiplier_ == 1) {
        result[row] = 0;
      }
      return;
    }

    bool value = bits::isBitSet(values, indices[row]);
    auto id = valueId(value);
    result[row] = multiplier_ == 1 ? id : result[row] + multiplier_ * id;
  });
  return true;
}

template <>
bool VectorHasher::makeValueIdsDecoded<bool, false>(
    const SelectivityVector& rows,
    uint64_t* result) {
  auto indices = decoded_.indices();
  auto values = decoded_.values<uint64_t>();

  rows.applyToSelected([&](vector_size_t row) INLINE_LAMBDA {
    bool value = bits::isBitSet(values, indices[row]);
    auto id = valueId(value);
    result[row] = multiplier_ == 1 ? id : result[row] + multiplier_ * id;
  });
  return true;
}

bool VectorHasher::computeValueIds(
    const BaseVector& values,
    const SelectivityVector& rows,
    raw_vector<uint64_t>& result) {
  decoded_.decode(values, rows);
  return VALUE_ID_TYPE_DISPATCH(makeValueIds, typeKind_, rows, result.data());
}

bool VectorHasher::computeValueIdsForRows(
    char** groups,
    int32_t numGroups,
    int32_t offset,
    int32_t nullByte,
    uint8_t nullMask,
    raw_vector<uint64_t>& result) {
  return VALUE_ID_TYPE_DISPATCH(
      makeValueIdsForRows,
      typeKind_,
      groups,
      numGroups,
      offset,
      nullByte,
      nullMask,
      result.data());
}

template <>
bool VectorHasher::makeValueIdsForRows<TypeKind::VARCHAR>(
    char** groups,
    int32_t numGroups,
    int32_t offset,
    int32_t nullByte,
    uint8_t nullMask,
    uint64_t* result) {
  for (int32_t i = 0; i < numGroups; ++i) {
    if (isNullAt(groups[i], nullByte, nullMask)) {
      if (multiplier_ == 1) {
        result[i] = 0;
      }
    } else {
      std::string storage;
      auto id = valueId<StringView>(HashStringAllocator::contiguousString(
          valueAt<StringView>(groups[i], offset), storage));
      if (id == kUnmappable) {
        return false;
      }
      result[i] = multiplier_ == 1 ? id : result[i] + multiplier_ * id;
    }
  }
  return true;
}

template <TypeKind Kind>
void VectorHasher::lookupValueIdsTyped(
    const DecodedVector& decoded,
    SelectivityVector& rows,
    raw_vector<uint64_t>& hashes,
    uint64_t* result) const {
  using T = typename TypeTraits<Kind>::NativeType;
  if (decoded.isConstantMapping()) {
    if (decoded.isNullAt(rows.begin())) {
      if (multiplier_ == 1) {
        rows.applyToSelected([&](vector_size_t row)
                                 INLINE_LAMBDA { result[row] = 0; });
      }
      return;
    }

    uint64_t id = lookupValueId(decoded.valueAt<T>(rows.begin()));
    if (id == kUnmappable) {
      rows.clearAll();
    } else {
      rows.applyToSelected([&](vector_size_t row) INLINE_LAMBDA {
        result[row] = multiplier_ == 1 ? id : result[row] + multiplier_ * id;
      });
    }
  } else if (decoded.isIdentityMapping()) {
    rows.applyToSelected([&](vector_size_t row) INLINE_LAMBDA {
      if (decoded.isNullAt(row)) {
        if (multiplier_ == 1) {
          result[row] = 0;
        }
        return;
      }
      T value = decoded.valueAt<T>(row);
      uint64_t id = lookupValueId(value);
      if (id == kUnmappable) {
        rows.setValid(row, false);
        return;
      }
      result[row] = multiplier_ == 1 ? id : result[row] + multiplier_ * id;
    });
    rows.updateBounds();
  } else {
    hashes.resize(decoded.base()->size());
    std::fill(hashes.begin(), hashes.end(), 0);
    rows.applyToSelected([&](vector_size_t row) INLINE_LAMBDA {
      if (decoded.isNullAt(row)) {
        if (multiplier_ == 1) {
          result[row] = 0;
        }
        return;
      }
      auto baseIndex = decoded.index(row);
      uint64_t id = hashes[baseIndex];
      if (id == 0) {
        T value = decoded.valueAt<T>(row);
        id = lookupValueId(value);
        if (id == kUnmappable) {
          rows.setValid(row, false);
          return;
        }
        hashes[baseIndex] = id;
      }
      result[row] = multiplier_ == 1 ? id : result[row] + multiplier_ * id;
    });
    rows.updateBounds();
  }
}

void VectorHasher::lookupValueIds(
    const BaseVector& values,
    SelectivityVector& rows,
    ScratchMemory& scratchMemory,
    raw_vector<uint64_t>& result) const {
  scratchMemory.decoded.decode(values, rows);
  VALUE_ID_TYPE_DISPATCH(
      lookupValueIdsTyped,
      typeKind_,
      scratchMemory.decoded,
      rows,
      scratchMemory.hashes,
      result.data());
}

void VectorHasher::hash(
    const BaseVector& values,
    const SelectivityVector& rows,
    bool mix,
    raw_vector<uint64_t>& result) {
  decoded_.decode(values, rows);
  return VELOX_DYNAMIC_TYPE_DISPATCH(
      hashValues, typeKind_, rows, mix, result.data());
}

void VectorHasher::analyze(
    char** groups,
    int32_t numGroups,
    int32_t offset,
    int32_t nullByte,
    uint8_t nullMask) {
  return VALUE_ID_TYPE_DISPATCH(
      analyzeTyped, typeKind_, groups, numGroups, offset, nullByte, nullMask);
}

template <>
void VectorHasher::analyzeValue(StringView value) {
  int size = value.size();
  auto data = value.data();
  if (!rangeOverflow_) {
    if (size > kStringASRangeMaxSize) {
      rangeOverflow_ = true;
    } else {
      int64_t number = stringAsNumber(data, size);
      updateRange(number);
    }
  }
  if (!distinctOverflow_) {
    UniqueValue unique(data, size);
    unique.setId(uniqueValues_.size() + 1);
    auto pair = uniqueValues_.insert(unique);
    if (pair.second) {
      if (uniqueValues_.size() > kMaxDistinct) {
        distinctOverflow_ = true;
        return;
      }
      copyStringToLocal(&*pair.first);
    }
  }
}

void VectorHasher::copyStringToLocal(const UniqueValue* unique) {
  auto size = unique->size();
  if (size <= sizeof(int64_t)) {
    return;
  }
  if (distinctStringsBytes_ > kMaxDistinctStringsBytes) {
    distinctOverflow_ = true;
    return;
  }
  if (uniqueValuesStorage_.empty()) {
    uniqueValuesStorage_.emplace_back();
    uniqueValuesStorage_.back().reserve(std::max(kStringBufferUnitSize, size));
    distinctStringsBytes_ += uniqueValuesStorage_.back().capacity();
  }
  auto str = &uniqueValuesStorage_.back();
  if (str->size() + size > str->capacity()) {
    uniqueValuesStorage_.emplace_back();
    uniqueValuesStorage_.back().reserve(std::max(kStringBufferUnitSize, size));
    distinctStringsBytes_ += uniqueValuesStorage_.back().capacity();
    str = &uniqueValuesStorage_.back();
  }
  auto start = str->size();
  str->resize(start + size);
  memcpy(str->data() + start, reinterpret_cast<char*>(unique->data()), size);
  const_cast<UniqueValue*>(unique)->setData(
      reinterpret_cast<int64_t>(str->data() + start));
}

std::unique_ptr<common::Filter> VectorHasher::getFilter(
    bool nullAllowed) const {
  switch (typeKind_) {
    case TypeKind::TINYINT:
    case TypeKind::SMALLINT:
    case TypeKind::INTEGER:
    case TypeKind::BIGINT: {
      if (!distinctOverflow_) {
        std::vector<int64_t> values;
        values.reserve(uniqueValues_.size());
        for (const auto& value : uniqueValues_) {
          values.emplace_back(value.data());
        }

        return common::createBigintValues(values, nullAllowed);
      }
    }
    default:
      // TODO Add support for strings.
      return nullptr;
  }
}

namespace {

V32::TV stringViewOffsets = {0, 4, 8, 12, 16, 20, 24, 28};

inline void updateResult(
    bool multiply,
    __m256i multiplier,
    __m256i values,
    uint64_t* result) {
  if (!multiply) {
    V64::store(result, values);
  } else {
    V64::store(result, V64::load(result) + values * multiplier);
  }
}

inline bool processPrefix64(
    __m256i prefixes,
    __m256i lengths,
    __m256i min,
    __m256i max,
    bool multiply,
    __m256i multiplier,
    uint64_t* result) {
  static int64_t lengthMasks64[8] = {
      0,
      1UL << 8,
      1UL << 16,
      1UL << 24,
      1UL << 32,
      1UL << 40,
      1UL << 48,
      1UL << 56};
  auto masks = V64::gather64<8>(lengthMasks64, lengths);
  prefixes |= masks;
  if (V64::compareResult(
          V64::compareGt(min, prefixes) | V64::compareGt(prefixes, max))) {
    return false;
  }
  auto numbers = prefixes - min + 1;
  updateResult(multiply, multiplier, numbers, result);
  return true;
}

inline __m256i
loadPrefixes64(const void* base, int64_t min, int32_t i, int32_t end) {
  auto start = reinterpret_cast<const char*>(base) + sizeof(StringView) * i + 4;
  auto indices = *reinterpret_cast<const __m128si_u*>(&stringViewOffsets);
  if (i + 4 <= end) {
    return V64::gather32<4>(start, indices);
  } else if (i >= end) {
    return V64::setAll(min);
  } else {
    return V64::maskGather32<4>(
        V64::setAll(min), V64::leadingMask(end - i), start, indices);
  }
}
} // namespace

template <>
bool VectorHasher::tryMapToRange(
    const StringView* values,
    const SelectivityVector& rows,
    uint64_t* result) {
  if (!FLAGS_enable_str_simd || !process::hasAvx2() || !rows.isAllSelected()) {
    return false;
  }
  auto end = rows.end();
  int32_t i = 0;
  auto min = V64::setAll(min_);
  auto max = V64::setAll(max_);
  __m256si lengthMasks32 = {0, 1 << 8, 1 << 16, 1 << 24, 0, 0, 0, 0};
  V64::TV multiplier = {0, 0, 0, 0};
  bool multiply = false;
  if (multiplier_ != 1) {
    multiply = true;
    multiplier = V64::setAll(multiplier_);
  }
  auto allZero = V32::setAll(0);
  // Points to the first of each batch of 8 consecutive StringViews.
  auto first = reinterpret_cast<const int32_t*>(values);
  for (; i < end; i += 8) {
    V32::TV lengths;
    V32::TV prefixes;
    if (UNLIKELY(i + 8 > end)) {
      auto minLength =
          V32::setAll(!min_ ? 0 : (63 - __builtin_clzll(min_)) / 8);
      auto mask = V32::leadingMask(end - i);
      lengths = V32::maskGather32(minLength, mask, first, stringViewOffsets);
    } else {
      lengths = V32::gather32(first, stringViewOffsets);
    }
    if (V32::compareResult(
            V32::compareGt(lengths, V32::setAll(rangeMaxChars_)))) {
      // At least one is longer than maximum allowed.
      return false;
    }
    if (rangeMaxChars_ <= 3 ||
        V32::kAllTrue ==
            V32::compareResult(V32::compareGt(V32::setAll(4), lengths))) {
      // All fit in 32 bits.
      if (i + 8 <= end) {
        prefixes = V32::gather32(first + 1, stringViewOffsets);
      } else {
        prefixes = V32::maskGather32(
            V32::setAll(min_),
            V32::leadingMask(end - i),
            first + 1,
            stringViewOffsets);
      }
      auto masks = reinterpret_cast<__m256si>(_mm256_permutevar8x32_epi32(
          simd::to256i(lengthMasks32), simd::to256i(lengths)));
      prefixes |= masks;
      if (V32::compareResult(
              V32::compareGt(V32::setAll(min_), prefixes) |
              V32::compareGt(prefixes, V32::setAll(max_)))) {
        return false;
      }
      prefixes -= static_cast<int32_t>(min_ - 1);
      updateResult(multiply, multiplier, V32::as4x64u<0>(prefixes), result + i);
      updateResult(
          multiply, multiplier, V32::as4x64u<1>(prefixes), result + i + 4);
    } else {
      if (!processPrefix64(
              loadPrefixes64(values, min_, i, end),
              V32::as4x64u<0>(lengths),
              min,
              max,
              multiply,
              multiplier,
              result + i)) {
        return false;
      }
      if (!processPrefix64(
              loadPrefixes64(values, min_, i + 4, end),
              V32::as4x64u<1>(lengths),
              min,
              max,
              multiply,
              multiplier,
              result + i + 4)) {
        return false;
      }
    }
    first += (8 * sizeof(StringView)) / sizeof(int32_t);
  }
  return true;
}

void VectorHasher::cardinality(uint64_t& asRange, uint64_t& asDistincts) {
  if (typeKind_ == TypeKind::BOOLEAN) {
    hasRange_ = true;
    asRange = 3;
    asDistincts = 3;
    return;
  }
  int64_t signedRange;
  if (!hasRange_ || rangeOverflow_) {
    asRange = kRangeTooLarge;
  } else if (__builtin_sub_overflow(max_, min_, &signedRange)) {
    rangeOverflow_ = true;
    asRange = kRangeTooLarge;
  } else if (signedRange < kMaxRange) {
    // If min is 10 and max is 20 then cardinality is 11 distinct
    // values in the closed interval + 1 for null.
    asRange = signedRange + 2;
  } else {
    rangeOverflow_ = true;
    asRange = kRangeTooLarge;
  }
  if (distinctOverflow_) {
    asDistincts = kRangeTooLarge;
    return;
  }
  // Count of values + 1 for null.
  asDistincts = uniqueValues_.size() + 1;
}

uint64_t VectorHasher::enableValueIds(uint64_t multiplier, int64_t reserve) {
  VELOX_CHECK_NE(
      typeKind_,
      TypeKind::BOOLEAN,
      "A boolean VectorHasher should  always be by range");
  multiplier_ = multiplier;
  rangeSize_ = uniqueValues_.size() + 1 + reserve;
  isRange_ = false;
  uint64_t result;
  if (__builtin_mul_overflow(multiplier_, rangeSize_, &result)) {
    return kRangeTooLarge;
  }
  return result;
}

uint64_t VectorHasher::enableValueRange(uint64_t multiplier, int64_t reserve) {
  static constexpr int64_t kMin = std::numeric_limits<int64_t>::min();
  static constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
  // Use reserve as padding above and below the range.
  reserve /= 2;
  multiplier_ = multiplier;
  VELOX_CHECK(hasRange_);
  if (kMin + reserve + 1 > min_) {
    min_ = kMin;
  } else {
    min_ -= reserve;
  }
  if (kMax - reserve < max_) {
    max_ = kMax;
  } else {
    max_ += reserve;
  }
  rangeMaxChars_ = max_ ? (64 - __builtin_clzll(max_)) / 8 : 0;
  isRange_ = true;
  // No overflow because max range is under 63 bits.
  if (typeKind_ == TypeKind::BOOLEAN) {
    rangeSize_ = 3;
  } else {
    rangeSize_ = (max_ - min_) + 2;
  }
  uint64_t result;
  if (__builtin_mul_overflow(multiplier_, rangeSize_, &result)) {
    return kRangeTooLarge;
  }
  return result;
}

void VectorHasher::merge(const VectorHasher& other) {
  if (typeKind_ == TypeKind::BOOLEAN) {
    return;
  }
  if (hasRange_ && other.hasRange_ && !rangeOverflow_ &&
      !other.rangeOverflow_) {
    min_ = std::min(min_, other.min_);
    max_ = std::max(max_, other.max_);
  } else {
    hasRange_ = false;
    rangeOverflow_ = true;
  }
  if (!distinctOverflow_ && !other.distinctOverflow_) {
    // Unique values can be merged without dispatch on type. All the
    // merged hashers must stay live for string type columns.
    for (UniqueValue value : other.uniqueValues_) {
      // Assign a new id at end of range for the case 'value' is not
      // in 'uniqueValues_'. We do not set overflow here because the
      // memory is already allocated and there is a known cap on size.
      value.setId(uniqueValues_.size() + 1);
      uniqueValues_.insert(value);
    }
  } else {
    distinctOverflow_ = true;
  }
}

std::string VectorHasher::toString() const {
  std::stringstream out;
  out << "<VectorHasher type=" << type_->toString() << "  isRange_=" << isRange_
      << " rangeSize= " << rangeSize_ << " min=" << min_ << " max=" << max_
      << " multiplier=" << multiplier_
      << " numDistinct=" << uniqueValues_.size() << ">";
  return out.str();
}

} // namespace facebook::velox::exec
