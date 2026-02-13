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
#include "velox/serializers/KeyEncoder.h"

#include "velox/common/base/SimdUtil.h"
#include "velox/type/Timestamp.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::serializer {
namespace {

bool validateBound(
    const IndexBound& indexBound,
    const std::vector<std::string>& indexColumns) {
  if (indexBound.bound == nullptr || indexBound.bound->size() == 0) {
    return false;
  }

  const auto& rowType = asRowType(indexBound.bound->type());
  if (rowType->size() != indexColumns.size()) {
    return false;
  }
  for (const auto& columnName : indexColumns) {
    if (!rowType->containsChild(columnName)) {
      return false;
    }
  }
  return true;
}

} // namespace

bool IndexBounds::validate() const {
  if (!lowerBound.has_value() && !upperBound.has_value()) {
    return false;
  }

  if (lowerBound.has_value() &&
      !validateBound(lowerBound.value(), indexColumns)) {
    return false;
  }

  if (upperBound.has_value() &&
      !validateBound(upperBound.value(), indexColumns)) {
    return false;
  }

  if (lowerBound.has_value() && upperBound.has_value()) {
    if (!lowerBound->bound->type()->equivalent(*upperBound->bound->type())) {
      return false;
    }
    if (lowerBound->bound->size() != upperBound->bound->size()) {
      return false;
    }
  }

  return true;
}

TypePtr IndexBounds::type() const {
  const RowVectorPtr& boundVector =
      lowerBound.has_value() ? lowerBound->bound : upperBound->bound;
  return boundVector->type();
}

std::string IndexBounds::toString() const {
  std::stringstream ss;
  ss << "IndexBounds{indexColumns=[";
  for (size_t i = 0; i < indexColumns.size(); ++i) {
    if (i > 0) {
      ss << ", ";
    }
    ss << indexColumns[i];
  }
  ss << "]";

  const auto formatBound =
      [&ss](const char* name, const std::optional<IndexBound>& bound) {
        if (!bound.has_value()) {
          ss << ", " << name << "=unbounded";
        } else {
          ss << ", " << name << "=" << (bound->inclusive ? "[" : "(");
          ss << bound->bound->toString(0, bound->bound->size());
          ss << (bound->inclusive ? "]" : ")");
        }
      };

  formatBound("lowerBound", lowerBound);
  formatBound("upperBound", upperBound);

  ss << "}";
  return ss.str();
}

vector_size_t IndexBounds::numRows() const {
  if (lowerBound.has_value()) {
    return lowerBound->bound->size();
  }
  VELOX_CHECK(upperBound.has_value());
  return upperBound->bound->size();
}

void IndexBounds::set(IndexBound lower, IndexBound upper) {
  lowerBound = std::move(lower);
  upperBound = std::move(upper);
  VELOX_CHECK(validate());
}

void IndexBounds::clear() {
  lowerBound.reset();
  upperBound.reset();
}

namespace {
// Validates if a type is a valid index column type.
// Only primitive scalar types are supported except UNKNOWN and HUGEINT.
bool isValidIndexColumnType(const TypePtr& type) {
  return type->isPrimitiveType() && type->kind() != TypeKind::UNKNOWN &&
      type->kind() != TypeKind::HUGEINT;
}

static constexpr int64_t DOUBLE_EXP_BIT_MASK = 0x7FF0000000000000L;
static constexpr int64_t DOUBLE_SIGNIF_BIT_MASK = 0x000FFFFFFFFFFFFFL;

static constexpr int32_t FLOAT_EXP_BIT_MASK = 0x7F800000;
static constexpr int32_t FLOAT_SIGNIF_BIT_MASK = 0x007FFFFF;

static constexpr size_t kNullByteSize = 1;

// Converts a double to its int64_t bit representation, normalizing all NaN
// values to a canonical quiet NaN (0x7ff8000000000000L). This ensures
// consistent key encoding since different NaN bit patterns should compare
// equal.
FOLLY_ALWAYS_INLINE int64_t doubleToLong(double value) {
  const int64_t* result = reinterpret_cast<const int64_t*>(&value);

  if (((*result & DOUBLE_EXP_BIT_MASK) == DOUBLE_EXP_BIT_MASK) &&
      (*result & DOUBLE_SIGNIF_BIT_MASK) != 0L) {
    return 0x7ff8000000000000L;
  }
  return *result;
}

// Converts a float to its int32_t bit representation, normalizing all NaN
// values to a canonical quiet NaN (0x7fc00000). This ensures consistent key
// encoding since different NaN bit patterns should compare equal.
FOLLY_ALWAYS_INLINE int32_t floatToInt(float value) {
  const int32_t* result = reinterpret_cast<const int32_t*>(&value);

  if (((*result & FLOAT_EXP_BIT_MASK) == FLOAT_EXP_BIT_MASK) &&
      (*result & FLOAT_SIGNIF_BIT_MASK) != 0L) {
    return 0x7fc00000;
  }
  return *result;
}

FOLLY_ALWAYS_INLINE void encodeByte(int8_t value, bool descending, char*& out) {
  if (descending) {
    *out = 0xff ^ value;
  } else {
    *out = value;
  }
  ++out;
}

// Template for encoding integers (signed and unsigned).
// For signed types: flips the sign bit for lexicographic ordering.
// Converts to big-endian and applies descending transformation if needed.
template <typename T>
FOLLY_ALWAYS_INLINE void encodeInt(T value, bool descending, char*& out) {
  using UnsignedT = std::make_unsigned_t<T>;

  // Flip sign bit for signed types for lexicographic ordering
  if constexpr (std::is_signed_v<T>) {
    constexpr int kSignBitShift = sizeof(T) * 8 - 1;
    value ^= 1ULL << kSignBitShift;
  }

  // Convert to big-endian
  value = folly::Endian::big(value);

  // Apply descending transformation if needed (branchless)
  // If descending is true, XOR with all 1s; if false, XOR with 0
  const UnsignedT mask = -static_cast<UnsignedT>(descending);
  value ^= mask;

  std::memcpy(out, &value, sizeof(value));
  out += sizeof(value);
}

FOLLY_ALWAYS_INLINE void
encodeString(const char* data, size_t size, bool descending, char*& out) {
  size_t offset = 0;
  constexpr size_t kBatchSize = xsimd::batch<uint8_t>::size;

  // Precompute XOR mask for descending transformation
  const uint8_t xorMask = descending ? 0xFF : 0x00;

  // Process in SIMD batches to find bytes needing escaping
  if (size >= kBatchSize) {
    const auto zero = xsimd::batch<uint8_t>::broadcast(0);
    const auto one = xsimd::batch<uint8_t>::broadcast(1);
    const auto xorBatch = xsimd::batch<uint8_t>::broadcast(xorMask);

    while (offset + kBatchSize <= size) {
      auto batch = xsimd::batch<uint8_t>::load_unaligned(
          reinterpret_cast<const uint8_t*>(data + offset));

      // Check which bytes need escaping (are 0 or 1)
      const auto needsEscape = (batch == zero) | (batch == one);
      const auto escapeMask = simd::toBitMask(needsEscape);

      if (escapeMask == 0) {
        // Fast path: no bytes need escaping, bulk copy the entire batch
        // Apply descending transformation branchlessly (no-op if xorBatch is
        // all zeros)
        batch = batch ^ xorBatch;
        batch.store_unaligned(reinterpret_cast<uint8_t*>(out));
        out += kBatchSize;
        offset += kBatchSize;
      } else {
        // Some bytes need escaping, process byte-by-byte for this batch
        for (size_t i = 0; i < kBatchSize; ++i) {
          const uint8_t byte = data[offset + i];
          if (byte == 0 || byte == 1) {
            encodeByte(1, descending, out);
            encodeByte(byte, descending, out);
          } else {
            encodeByte(byte, descending, out);
          }
        }
        offset += kBatchSize;
      }
    }
  }

  // Process remaining bytes
  for (; offset < size; ++offset) {
    const uint8_t byte = data[offset];
    if (byte == 0 || byte == 1) {
      encodeByte(1, descending, out);
      encodeByte(byte, descending, out);
    } else {
      encodeByte(byte, descending, out);
    }
  }

  // Write terminator
  encodeByte(0, descending, out);
}

FOLLY_ALWAYS_INLINE size_t stringEncodedSize(const char* data, size_t size) {
  size_t countZeros{0};
  size_t countOnes{0};
  size_t offset{0};

  // Process bytes in SIMD batches
  constexpr size_t kBatchSize = xsimd::batch<uint8_t>::size;
  if (size >= kBatchSize) {
    const auto zero = xsimd::batch<uint8_t>::broadcast(0);
    const auto one = xsimd::batch<uint8_t>::broadcast(1);

    for (; offset + kBatchSize <= size; offset += kBatchSize) {
      auto batch = xsimd::batch<uint8_t>::load_unaligned(
          reinterpret_cast<const uint8_t*>(data + offset));

      // Count bytes that equal 0
      const auto isZero = (batch == zero);
      const auto zeroMask = simd::toBitMask(isZero);
      countZeros += __builtin_popcount(zeroMask);

      // Count bytes that equal 1
      const auto isOne = (batch == one);
      const auto oneMask = simd::toBitMask(isOne);
      countOnes += __builtin_popcount(oneMask);
    }
  }

  // Process remaining bytes using standard algorithms
  const auto* remaining = reinterpret_cast<const uint8_t*>(data + offset);
  const auto remainingSize = size - offset;

  countZeros += std::count(remaining, remaining + remainingSize, 0);
  countOnes += std::count(remaining, remaining + remainingSize, 1);

  // Each 0 or 1 byte needs 2 bytes (escape byte + actual byte)
  // Other bytes need 1 byte
  // Plus 1 terminator byte
  return size + countZeros + countOnes + 1;
}

FOLLY_ALWAYS_INLINE void encodeBool(bool value, char*& out) {
  encodeByte(value, /*descending=*/false, out);
}

// Forward declarations for functions used in estimateEncodedColumnSize
void addDateEncodedSize(
    const folly::Range<const vector_size_t*>& nonNullRows,
    vector_size_t** nonNullSizePtrs);

template <TypeKind KIND>
void addColumnEncodedSizeTyped(
    const DecodedVector& decodedSource,
    const folly::Range<const vector_size_t*>& nonNullRows,
    vector_size_t** nonNullSizePtrs);

void estimateEncodedColumnSize(
    const DecodedVector& decodedSource,
    const folly::Range<const vector_size_t*>& rows,
    std::vector<vector_size_t>& sizes,
    Scratch& scratch) {
  const auto numRows = rows.size();
  folly::Range<const vector_size_t*> nonNullRows = rows;
  vector_size_t** nonNullSizePtrs = nullptr;

  ScratchPtr<uint64_t, 64> nullsHolder(scratch);
  ScratchPtr<vector_size_t, 64> nonNullsHolder(scratch);
  ScratchPtr<vector_size_t*, 64> nonNullSizePtrsHolder(scratch);

  if (decodedSource.mayHaveNulls()) {
    auto* nulls = nullsHolder.get(bits::nwords(numRows));
    for (auto row = 0; row < numRows; ++row) {
      bits::clearNull(nulls, row);
      if (decodedSource.isNullAt(rows[row])) {
        bits::setNull(nulls, row);
        sizes[row] += kNullByteSize;
      }
    }

    auto* nonNulls = nonNullsHolder.get(numRows);
    const auto numNonNull = simd::indicesOfSetBits(nulls, 0, numRows, nonNulls);
    if (numNonNull == 0) {
      return;
    }

    auto* const nonNullSizes = sizes.data();
    nonNullSizePtrs = nonNullSizePtrsHolder.get(numNonNull);
    for (int32_t i = 0; i < numNonNull; ++i) {
      nonNullSizePtrs[i] = &nonNullSizes[nonNulls[i]];
    }

    simd::transpose(
        rows.data(),
        folly::Range<const vector_size_t*>(nonNulls, numNonNull),
        nonNulls);
    nonNullRows = folly::Range<const vector_size_t*>(nonNulls, numNonNull);
  } else {
    // No nulls, all rows are non-null - create size pointers for all rows
    nonNullSizePtrs = nonNullSizePtrsHolder.get(numRows);
    for (int32_t i = 0; i < numRows; ++i) {
      nonNullSizePtrs[i] = &sizes[i];
    }
  }

  // Process data for non-null rows only (or all rows if no nulls)
  if (decodedSource.base()->type()->isDate()) {
    for (auto i = 0; i < nonNullRows.size(); ++i) {
      *nonNullSizePtrs[i] += (kNullByteSize + sizeof(int32_t));
    }
  } else {
    VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
        addColumnEncodedSizeTyped,
        decodedSource.base()->typeKind(),
        decodedSource,
        nonNullRows,
        nonNullSizePtrs);
  }
}

template <TypeKind KIND>
void addColumnEncodedSizeTyped(
    const DecodedVector& decodedSource,
    const folly::Range<const vector_size_t*>& nonNullRows,
    vector_size_t** nonNullSizePtrs) {
  // Handle variable-width types
  if constexpr (KIND == TypeKind::VARCHAR || KIND == TypeKind::VARBINARY) {
    for (auto i = 0; i < nonNullRows.size(); ++i) {
      const auto view = decodedSource.valueAt<StringView>(nonNullRows[i]);
      *nonNullSizePtrs[i] +=
          kNullByteSize + stringEncodedSize(view.data(), view.size());
    }
  } else {
    // Fixed-width scalar types
    const auto elementSize = decodedSource.base()->type()->cppSizeInBytes();
    for (auto i = 0; i < nonNullRows.size(); ++i) {
      *nonNullSizePtrs[i] += (kNullByteSize + elementSize);
    }
  }
}

std::vector<vector_size_t> getKeyChannels(
    const RowTypePtr& inputType,
    const std::vector<std::string>& keyColumns) {
  std::vector<vector_size_t> keyChannels;
  keyChannels.reserve(keyColumns.size());
  for (const auto& keyColumn : keyColumns) {
    keyChannels.emplace_back(inputType->getChildIdx(keyColumn));
  }
  return keyChannels;
}

// Wrapper functions for backward compatibility and readability
FOLLY_ALWAYS_INLINE void
encodeLong(int64_t value, bool descending, char*& out) {
  encodeInt(value, descending, out);
}

FOLLY_ALWAYS_INLINE void
encodeUnsignedLong(uint64_t value, bool descending, char*& out) {
  encodeInt(value, descending, out);
}

FOLLY_ALWAYS_INLINE void
encodeShort(int16_t value, bool descending, char*& out) {
  encodeInt(value, descending, out);
}

FOLLY_ALWAYS_INLINE void
encodeInteger(int32_t value, bool descending, char*& out) {
  encodeInt(value, descending, out);
}

FOLLY_ALWAYS_INLINE void
encodeUnsignedInteger(uint32_t value, bool descending, char*& out) {
  encodeInt(value, descending, out);
}

void encodeBigInt(
    const DecodedVector& decodedVector,
    vector_size_t numRows,
    bool descending,
    bool nullLast,
    std::vector<char*>& rowOffsets) {
  const bool mayHaveNulls = decodedVector.mayHaveNulls();
  if (mayHaveNulls) {
    for (auto row = 0; row < numRows; ++row) {
      if (decodedVector.isNullAt(row)) {
        // Encode null indicator only
        encodeBool(nullLast, rowOffsets[row]);
      } else {
        // Encode non-null indicator followed by data
        encodeBool(!nullLast, rowOffsets[row]);
        const auto value = decodedVector.valueAt<int64_t>(row);
        encodeLong(value, descending, rowOffsets[row]);
      }
    }
  } else {
    for (auto row = 0; row < numRows; ++row) {
      // Encode non-null indicator followed by data
      encodeBool(!nullLast, rowOffsets[row]);
      const auto value = decodedVector.valueAt<int64_t>(row);
      encodeLong(value, descending, rowOffsets[row]);
    }
  }
}

void encodeBoolean(
    const DecodedVector& decodedVector,
    vector_size_t numRows,
    bool descending,
    bool nullLast,
    std::vector<char*>& rowOffsets) {
  const bool mayHaveNulls = decodedVector.mayHaveNulls();
  if (mayHaveNulls) {
    for (auto row = 0; row < numRows; ++row) {
      if (decodedVector.isNullAt(row)) {
        // Encode null indicator only
        encodeBool(nullLast, rowOffsets[row]);
      } else {
        // Encode non-null indicator followed by data
        encodeBool(!nullLast, rowOffsets[row]);
        const auto value = decodedVector.valueAt<bool>(row);
        encodeByte(
            static_cast<int8_t>(value ? 2 : 1), descending, rowOffsets[row]);
      }
    }
  } else {
    for (auto row = 0; row < numRows; ++row) {
      // Encode non-null indicator followed by data
      encodeBool(!nullLast, rowOffsets[row]);
      const auto value = decodedVector.valueAt<bool>(row);
      encodeByte(
          static_cast<int8_t>(value ? 2 : 1), descending, rowOffsets[row]);
    }
  }
}

void encodeDouble(
    const DecodedVector& decodedVector,
    vector_size_t numRows,
    bool descending,
    bool nullLast,
    std::vector<char*>& rowOffsets) {
  const bool mayHaveNulls = decodedVector.mayHaveNulls();
  if (mayHaveNulls) {
    for (auto row = 0; row < numRows; ++row) {
      if (decodedVector.isNullAt(row)) {
        // Encode null indicator only
        encodeBool(nullLast, rowOffsets[row]);
      } else {
        // Encode non-null indicator followed by data
        encodeBool(!nullLast, rowOffsets[row]);
        const auto value = decodedVector.valueAt<double>(row);
        int64_t longValue = doubleToLong(value);
        // Normalize -0.0 to +0.0 by clearing sign bit when value is zero
        if ((longValue & ~(1L << 63)) == 0) {
          longValue = 0;
        }
        if ((longValue & (1L << 63)) != 0) {
          longValue = ~longValue;
        } else {
          longValue = longValue ^ (1L << 63);
        }
        encodeUnsignedLong(longValue, descending, rowOffsets[row]);
      }
    }
  } else {
    for (auto row = 0; row < numRows; ++row) {
      // Encode non-null indicator followed by data
      encodeBool(!nullLast, rowOffsets[row]);
      const auto value = decodedVector.valueAt<double>(row);
      int64_t longValue = doubleToLong(value);
      // Normalize -0.0 to +0.0 by clearing sign bit when value is zero
      if ((longValue & ~(1L << 63)) == 0) {
        longValue = 0;
      }
      if ((longValue & (1L << 63)) != 0) {
        longValue = ~longValue;
      } else {
        longValue = longValue ^ (1L << 63);
      }
      encodeUnsignedLong(longValue, descending, rowOffsets[row]);
    }
  }
}

void encodeReal(
    const DecodedVector& decodedVector,
    vector_size_t numRows,
    bool descending,
    bool nullLast,
    std::vector<char*>& rowOffsets) {
  const bool mayHaveNulls = decodedVector.mayHaveNulls();
  if (mayHaveNulls) {
    for (auto row = 0; row < numRows; ++row) {
      if (decodedVector.isNullAt(row)) {
        // Encode null indicator only
        encodeBool(nullLast, rowOffsets[row]);
      } else {
        // Encode non-null indicator followed by data
        encodeBool(!nullLast, rowOffsets[row]);
        const auto value = decodedVector.valueAt<float>(row);
        int32_t intValue = floatToInt(value);
        // Normalize -0.0 to +0.0 by clearing sign bit when value is zero
        if ((intValue & ~(1 << 31)) == 0) {
          intValue = 0;
        }
        if ((intValue & (1L << 31)) != 0) {
          intValue = ~intValue;
        } else {
          intValue = intValue ^ (1L << 31);
        }
        encodeUnsignedInteger(intValue, descending, rowOffsets[row]);
      }
    }
  } else {
    for (auto row = 0; row < numRows; ++row) {
      // Encode non-null indicator followed by data
      encodeBool(!nullLast, rowOffsets[row]);
      const auto value = decodedVector.valueAt<float>(row);
      int32_t intValue = floatToInt(value);
      // Normalize -0.0 to +0.0 by clearing sign bit when value is zero
      if ((intValue & ~(1 << 31)) == 0) {
        intValue = 0;
      }
      if ((intValue & (1L << 31)) != 0) {
        intValue = ~intValue;
      } else {
        intValue = intValue ^ (1L << 31);
      }
      encodeUnsignedInteger(intValue, descending, rowOffsets[row]);
    }
  }
}

void encodeTinyInt(
    const DecodedVector& decodedVector,
    vector_size_t numRows,
    bool descending,
    bool nullLast,
    std::vector<char*>& rowOffsets) {
  const bool mayHaveNulls = decodedVector.mayHaveNulls();
  if (mayHaveNulls) {
    for (auto row = 0; row < numRows; ++row) {
      if (decodedVector.isNullAt(row)) {
        // Encode null indicator only
        encodeBool(nullLast, rowOffsets[row]);
      } else {
        // Encode non-null indicator followed by data
        encodeBool(!nullLast, rowOffsets[row]);
        const auto value = decodedVector.valueAt<int8_t>(row);
        encodeByte(
            static_cast<int8_t>(value ^ 0x80), descending, rowOffsets[row]);
      }
    }
  } else {
    for (auto row = 0; row < numRows; ++row) {
      // Encode non-null indicator followed by data
      encodeBool(!nullLast, rowOffsets[row]);
      const auto value = decodedVector.valueAt<int8_t>(row);
      encodeByte(
          static_cast<int8_t>(value ^ 0x80), descending, rowOffsets[row]);
    }
  }
}

void encodeSmallInt(
    const DecodedVector& decodedVector,
    vector_size_t numRows,
    bool descending,
    bool nullLast,
    std::vector<char*>& rowOffsets) {
  const bool mayHaveNulls = decodedVector.mayHaveNulls();
  if (mayHaveNulls) {
    for (auto row = 0; row < numRows; ++row) {
      if (decodedVector.isNullAt(row)) {
        // Encode null indicator only
        encodeBool(nullLast, rowOffsets[row]);
      } else {
        // Encode non-null indicator followed by data
        encodeBool(!nullLast, rowOffsets[row]);
        const auto value = decodedVector.valueAt<int16_t>(row);
        encodeShort(value, descending, rowOffsets[row]);
      }
    }
  } else {
    for (auto row = 0; row < numRows; ++row) {
      // Encode non-null indicator followed by data
      encodeBool(!nullLast, rowOffsets[row]);
      const auto value = decodedVector.valueAt<int16_t>(row);
      encodeShort(value, descending, rowOffsets[row]);
    }
  }
}

void encodeIntegerType(
    const DecodedVector& decodedVector,
    vector_size_t numRows,
    bool descending,
    bool nullLast,
    std::vector<char*>& rowOffsets) {
  const bool mayHaveNulls = decodedVector.mayHaveNulls();
  if (mayHaveNulls) {
    for (auto row = 0; row < numRows; ++row) {
      if (decodedVector.isNullAt(row)) {
        // Encode null indicator only
        encodeBool(nullLast, rowOffsets[row]);
      } else {
        // Encode non-null indicator followed by data
        encodeBool(!nullLast, rowOffsets[row]);
        const auto value = decodedVector.valueAt<int32_t>(row);
        encodeInteger(value, descending, rowOffsets[row]);
      }
    }
  } else {
    for (auto row = 0; row < numRows; ++row) {
      // Encode non-null indicator followed by data
      encodeBool(!nullLast, rowOffsets[row]);
      const auto value = decodedVector.valueAt<int32_t>(row);
      encodeInteger(value, descending, rowOffsets[row]);
    }
  }
}

void encodeStringType(
    const DecodedVector& decodedVector,
    vector_size_t numRows,
    bool descending,
    bool nullLast,
    std::vector<char*>& rowOffsets) {
  const bool mayHaveNulls = decodedVector.mayHaveNulls();
  if (mayHaveNulls) {
    for (auto row = 0; row < numRows; ++row) {
      if (decodedVector.isNullAt(row)) {
        // Encode null indicator only
        encodeBool(nullLast, rowOffsets[row]);
      } else {
        // Encode non-null indicator followed by data
        encodeBool(!nullLast, rowOffsets[row]);
        const auto value = decodedVector.valueAt<StringView>(row);
        encodeString(value.data(), value.size(), descending, rowOffsets[row]);
      }
    }
  } else {
    for (auto row = 0; row < numRows; ++row) {
      // Encode non-null indicator followed by data
      encodeBool(!nullLast, rowOffsets[row]);
      const auto value = decodedVector.valueAt<StringView>(row);
      encodeString(value.data(), value.size(), descending, rowOffsets[row]);
    }
  }
}

// Encodes Timestamp type column for all rows in columnar fashion.
// When comparing Timestamp, first compare seconds and then compare nanos, so
// when encoding, just encode seconds and nanos in sequence (similar to
// PrefixSortEncoder).
void encodeTimestamp(
    const DecodedVector& decodedVector,
    vector_size_t numRows,
    bool descending,
    bool nullLast,
    std::vector<char*>& rowOffsets) {
  const bool mayHaveNulls = decodedVector.mayHaveNulls();
  if (mayHaveNulls) {
    for (auto row = 0; row < numRows; ++row) {
      if (decodedVector.isNullAt(row)) {
        // Encode null indicator only
        encodeBool(nullLast, rowOffsets[row]);
      } else {
        // Encode non-null indicator followed by data
        encodeBool(!nullLast, rowOffsets[row]);
        const auto value = decodedVector.valueAt<Timestamp>(row);
        // Encode seconds (int64_t) followed by nanos (uint64_t)
        encodeLong(value.getSeconds(), descending, rowOffsets[row]);
        encodeUnsignedLong(value.getNanos(), descending, rowOffsets[row]);
      }
    }
  } else {
    for (auto row = 0; row < numRows; ++row) {
      // Encode non-null indicator followed by data
      encodeBool(!nullLast, rowOffsets[row]);
      const auto value = decodedVector.valueAt<Timestamp>(row);
      // Encode seconds (int64_t) followed by nanos (uint64_t)
      encodeLong(value.getSeconds(), descending, rowOffsets[row]);
      encodeUnsignedLong(value.getNanos(), descending, rowOffsets[row]);
    }
  }
}

// Encodes Date type column for all rows in columnar fashion.
void encodeDate(
    const DecodedVector& decodedVector,
    vector_size_t numRows,
    bool descending,
    bool nullLast,
    std::vector<char*>& rowOffsets) {
  const bool mayHaveNulls = decodedVector.mayHaveNulls();
  if (mayHaveNulls) {
    for (auto row = 0; row < numRows; ++row) {
      if (decodedVector.isNullAt(row)) {
        // Encode null indicator only
        encodeBool(nullLast, rowOffsets[row]);
      } else {
        // Encode non-null indicator followed by data
        encodeBool(!nullLast, rowOffsets[row]);
        const auto value = decodedVector.valueAt<int32_t>(row);
        encodeInteger(value, descending, rowOffsets[row]);
      }
    }
  } else {
    for (auto row = 0; row < numRows; ++row) {
      // Encode non-null indicator followed by data
      encodeBool(!nullLast, rowOffsets[row]);
      const auto value = decodedVector.valueAt<int32_t>(row);
      encodeInteger(value, descending, rowOffsets[row]);
    }
  }
}

// Template function to encode column data for a specific type kind
template <TypeKind Kind>
void encodeColumnTyped(
    const DecodedVector& decodedVector,
    vector_size_t numRows,
    bool descending,
    bool nullLast,
    std::vector<char*>& rowOffsets) {
  if constexpr (Kind == TypeKind::BIGINT) {
    encodeBigInt(decodedVector, numRows, descending, nullLast, rowOffsets);
  } else if constexpr (Kind == TypeKind::BOOLEAN) {
    encodeBoolean(decodedVector, numRows, descending, nullLast, rowOffsets);
  } else if constexpr (Kind == TypeKind::DOUBLE) {
    encodeDouble(decodedVector, numRows, descending, nullLast, rowOffsets);
  } else if constexpr (Kind == TypeKind::REAL) {
    encodeReal(decodedVector, numRows, descending, nullLast, rowOffsets);
  } else if constexpr (Kind == TypeKind::TINYINT) {
    encodeTinyInt(decodedVector, numRows, descending, nullLast, rowOffsets);
  } else if constexpr (Kind == TypeKind::SMALLINT) {
    encodeSmallInt(decodedVector, numRows, descending, nullLast, rowOffsets);
  } else if constexpr (Kind == TypeKind::INTEGER) {
    encodeIntegerType(decodedVector, numRows, descending, nullLast, rowOffsets);
  } else if constexpr (Kind == TypeKind::TIMESTAMP) {
    encodeTimestamp(decodedVector, numRows, descending, nullLast, rowOffsets);
  } else if constexpr (
      Kind == TypeKind::VARCHAR || Kind == TypeKind::VARBINARY) {
    encodeStringType(decodedVector, numRows, descending, nullLast, rowOffsets);
  } else {
    VELOX_UNSUPPORTED("Unsupported type: {}", Kind);
  }
}

bool incrementStringValue(std::string* value, bool descending) {
  if (!descending) {
    // Ascending order: append the smallest possible byte (null byte) to get the
    // immediate next string. For any string s, s < s + '\0' < s + '\x01' < ...
    value->push_back('\0');
    return true;
  }
  // Descending order: decrement is only possible if the string ends with
  // '\0'. This is the inverse of the increment operation - truncate the
  // trailing null byte. This approach aligns with Apache Kudu's
  // DecrementStringCell.
  //
  // For strings not ending with '\0', there is no finite immediate
  // predecessor (e.g., predecessor of "abc" would be "ab\xFF\xFF\xFF..." with
  // infinite 0xFF bytes). Return false to signal caller should use exclusive
  // bounds.
  if (value->empty() || value->back() != '\0') {
    return false;
  }
  value->pop_back();
  return true;
}

template <TypeKind KIND>
bool setMinValueTyped(VectorPtr& result, vector_size_t row) {
  using T = typename TypeTraits<KIND>::NativeType;

  if constexpr (KIND == TypeKind::BOOLEAN) {
    auto* flatVector = result->asChecked<FlatVector<bool>>();
    flatVector->set(row, false);
    return true;
  }

  if constexpr (
      KIND == TypeKind::TINYINT || KIND == TypeKind::SMALLINT ||
      KIND == TypeKind::INTEGER || KIND == TypeKind::BIGINT) {
    auto* flatVector = result->asChecked<FlatVector<T>>();
    flatVector->set(row, std::numeric_limits<T>::min());
    return true;
  }

  if constexpr (KIND == TypeKind::REAL || KIND == TypeKind::DOUBLE) {
    auto* flatVector = result->asChecked<FlatVector<T>>();
    flatVector->set(row, -std::numeric_limits<T>::infinity());
    return true;
  }

  if constexpr (KIND == TypeKind::VARCHAR || KIND == TypeKind::VARBINARY) {
    auto* flatVector = result->asChecked<FlatVector<StringView>>();
    flatVector->set(row, StringView(""));
    return true;
  }

  if constexpr (KIND == TypeKind::TIMESTAMP) {
    auto* flatVector = result->asChecked<FlatVector<Timestamp>>();
    flatVector->set(row, std::numeric_limits<Timestamp>::min());
    return true;
  }

  VELOX_UNSUPPORTED("Cannot set min value for column type: {}", KIND);
}

template <TypeKind KIND>
bool setMaxValueTyped(VectorPtr& result, vector_size_t row) {
  using T = typename TypeTraits<KIND>::NativeType;

  if constexpr (KIND == TypeKind::BOOLEAN) {
    auto* flatVector = result->asChecked<FlatVector<bool>>();
    flatVector->set(row, true);
    return true;
  }

  if constexpr (
      KIND == TypeKind::TINYINT || KIND == TypeKind::SMALLINT ||
      KIND == TypeKind::INTEGER || KIND == TypeKind::BIGINT) {
    auto* flatVector = result->asChecked<FlatVector<T>>();
    flatVector->set(row, std::numeric_limits<T>::max());
    return true;
  }

  if constexpr (KIND == TypeKind::REAL || KIND == TypeKind::DOUBLE) {
    auto* flatVector = result->asChecked<FlatVector<T>>();
    flatVector->set(row, std::numeric_limits<T>::infinity());
    return true;
  }

  if constexpr (KIND == TypeKind::TIMESTAMP) {
    auto* flatVector = result->asChecked<FlatVector<Timestamp>>();
    flatVector->set(row, std::numeric_limits<Timestamp>::max());
    return true;
  }

  if constexpr (KIND == TypeKind::VARCHAR || KIND == TypeKind::VARBINARY) {
    // For max string value, we can't represent it directly, but we can use
    // a very large string. However, this is problematic. For now, return false.
    return false;
  }

  VELOX_UNSUPPORTED("Cannot set max value for column type: {}", KIND);
}

template <TypeKind KIND>
bool incrementColumnValueTyped(
    const VectorPtr& column,
    vector_size_t row,
    VectorPtr& result,
    bool descending) {
  using T = typename TypeTraits<KIND>::NativeType;

  if constexpr (KIND == TypeKind::BOOLEAN) {
    const auto* inputVector = column->asChecked<FlatVector<bool>>();
    auto* resultVector = result->asChecked<FlatVector<bool>>();
    const auto value = inputVector->valueAt(row);
    if (!descending) {
      // Ascending: false -> true, true cannot increment
      if (value) {
        // Overflow: wrap around to min value (false) for proper bound encoding.
        resultVector->set(row, false);
        return false;
      }
      resultVector->set(row, true);
    } else {
      // Descending: true -> false, false cannot decrement
      if (!value) {
        // Underflow: wrap around to max value (true) for proper bound encoding.
        resultVector->set(row, true);
        return false;
      }
      resultVector->set(row, false);
    }
    return true;
  }

  if constexpr (
      KIND == TypeKind::TINYINT || KIND == TypeKind::SMALLINT ||
      KIND == TypeKind::INTEGER || KIND == TypeKind::BIGINT) {
    const auto* inputVector = column->asChecked<FlatVector<T>>();
    auto* resultVector = result->asChecked<FlatVector<T>>();
    const auto value = inputVector->valueAt(row);
    if (!descending) {
      // Ascending: increment
      if (value == std::numeric_limits<T>::max()) {
        // Overflow: wrap around to min value for proper bound encoding.
        resultVector->set(row, std::numeric_limits<T>::min());
        return false;
      }
      resultVector->set(row, value + 1);
    } else {
      // Descending: decrement
      if (value == std::numeric_limits<T>::min()) {
        // Underflow: wrap around to max value for proper bound encoding.
        resultVector->set(row, std::numeric_limits<T>::max());
        return false;
      }
      resultVector->set(row, value - 1);
    }
    return true;
  }

  if constexpr (KIND == TypeKind::REAL || KIND == TypeKind::DOUBLE) {
    const auto* inputVector = column->asChecked<FlatVector<T>>();
    auto* resultVector = result->asChecked<FlatVector<T>>();
    const auto value = inputVector->valueAt(row);
    if (!descending) {
      // Ascending: increment to next representable value
      if (std::isinf(value) && value > 0) {
        // Overflow: wrap around to min value (-inf) for proper bound encoding.
        resultVector->set(row, -std::numeric_limits<T>::infinity());
        return false;
      }
      resultVector->set(
          row, std::nextafter(value, std::numeric_limits<T>::infinity()));
    } else {
      // Descending: decrement to previous representable value
      if (std::isinf(value) && value < 0) {
        // Underflow: wrap around to max value (+inf) for proper bound encoding.
        resultVector->set(row, std::numeric_limits<T>::infinity());
        return false;
      }
      resultVector->set(
          row, std::nextafter(value, -std::numeric_limits<T>::infinity()));
    }
    return true;
  }

  if constexpr (KIND == TypeKind::VARCHAR || KIND == TypeKind::VARBINARY) {
    // For strings, increment/decrement the string representation.
    const auto* inputVector = column->asChecked<FlatVector<StringView>>();
    auto* resultVector = result->asChecked<FlatVector<StringView>>();
    const auto value = inputVector->valueAt(row);
    std::string incrementedStr(value.data(), value.size());
    if (!incrementStringValue(&incrementedStr, descending)) {
      VELOX_CHECK(descending);
      // Descending underflow: should not happen in practice as VARCHAR
      // filter conversion is not supported for descending order.
      VELOX_UNREACHABLE(
          "Unexpected string underflow during descending bound increment");
    }
    resultVector->set(row, StringView(incrementedStr));
    return true;
  }

  if constexpr (KIND == TypeKind::TIMESTAMP) {
    const auto* inputVector = column->asChecked<FlatVector<Timestamp>>();
    auto* resultVector = result->asChecked<FlatVector<Timestamp>>();
    const auto value = inputVector->valueAt(row);
    if (!descending) {
      // Ascending: increment nanos first, then seconds if nanos overflow
      const auto nanos = value.getNanos();
      VELOX_DCHECK_LE(nanos, Timestamp::kMaxNanos);
      const auto seconds = value.getSeconds();
      VELOX_DCHECK_LE(seconds, Timestamp::kMaxSeconds);
      if (nanos < Timestamp::kMaxNanos) {
        resultVector->set(row, Timestamp(seconds, nanos + 1));
      } else if (seconds < Timestamp::kMaxSeconds) {
        resultVector->set(row, Timestamp(seconds + 1, 0));
      } else {
        // Overflow: wrap around to min timestamp for proper bound encoding.
        resultVector->set(row, Timestamp(Timestamp::kMinSeconds, 0));
        return false;
      }
    } else {
      // Descending: decrement nanos first, then seconds if nanos underflow
      const auto nanos = value.getNanos();
      const auto seconds = value.getSeconds();
      if (nanos > 0) {
        resultVector->set(row, Timestamp(seconds, nanos - 1));
      } else if (seconds > Timestamp::kMinSeconds) {
        resultVector->set(row, Timestamp(seconds - 1, Timestamp::kMaxNanos));
      } else {
        // Underflow: wrap around to max timestamp for proper bound encoding.
        resultVector->set(
            row, Timestamp(Timestamp::kMaxSeconds, Timestamp::kMaxNanos));
        return false;
      }
    }
    return true;
  }

  VELOX_UNSUPPORTED("Cannot increment column type: {}", KIND);
}

// Handles incrementing or decrementing null column values based on sort order.
// Returns true if operation succeeds, false otherwise.
bool incrementNullColumnValue(
    const VectorPtr& column,
    vector_size_t row,
    bool descending,
    bool nullLast,
    VectorPtr& result) {
  // For descending order, "next" value after null (or min in sort order) is
  // MAX. For ascending order, "next" value after null (or min in sort order) is
  // MIN.
  result->setNull(row, false);
  const auto ret = descending
      ? VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
            setMaxValueTyped, column->typeKind(), result, row)
      : VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
            setMinValueTyped, column->typeKind(), result, row);

  VELOX_CHECK(
      ret,
      "Cannot set {} value for type {} when incrementing NULL",
      descending ? "max" : "min",
      column->type()->toString());

  // Nulls first: increment succeeded, return true.
  // Nulls last: null is at end of sort order, reset for carry, return false.
  return !nullLast;
}

// Increments or decrements a column value at the given row index based on sort
// order. Returns true if increment/decrement succeeds, false if value
// overflows/underflows.
bool incrementColumnValue(
    const VectorPtr& column,
    vector_size_t row,
    bool descending,
    bool nullLast,
    VectorPtr& result) {
  if (column->isNullAt(row)) {
    return incrementNullColumnValue(column, row, descending, nullLast, result);
  }

  return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
      incrementColumnValueTyped,
      column->typeKind(),
      column,
      row,
      result,
      descending);
}
} // namespace

// static.
std::unique_ptr<KeyEncoder> KeyEncoder::create(
    std::vector<std::string> keyColumns,
    RowTypePtr inputType,
    std::vector<core::SortOrder> sortOrders,
    memory::MemoryPool* pool) {
  return std::unique_ptr<KeyEncoder>(new KeyEncoder(
      std::move(keyColumns),
      std::move(inputType),
      std::move(sortOrders),
      pool));
}

KeyEncoder::KeyEncoder(
    std::vector<std::string> keyColumns,
    RowTypePtr inputType,
    std::vector<core::SortOrder> sortOrders,
    memory::MemoryPool* pool)
    : inputType_{std::move(inputType)},
      sortOrders_{std::move(sortOrders)},
      keyChannels_{getKeyChannels(inputType_, keyColumns)},
      pool_{pool},
      childDecodedVectors_{keyColumns.size()} {
  VELOX_CHECK(!childDecodedVectors_.empty());
  VELOX_CHECK_EQ(
      keyChannels_.size(),
      sortOrders_.size(),
      "Size mismatch between key columns and sort orders");

  // Validate that all index columns are valid types
  for (size_t i = 0; i < keyChannels_.size(); ++i) {
    const auto& columnType = inputType_->childAt(keyChannels_[i]);
    VELOX_CHECK(
        isValidIndexColumnType(columnType),
        "Unsupported type for index column '{}': {}",
        keyColumns[i],
        columnType->toString());
  }
}

void KeyEncoder::encodeColumn(
    const DecodedVector& decodedVector,
    vector_size_t numRows,
    bool descending,
    bool nullLast,
    std::vector<char*>& rowOffsets) {
  const auto typeKind = decodedVector.base()->typeKind();
  if (decodedVector.base()->type()->isDate()) {
    encodeDate(decodedVector, numRows, descending, nullLast, rowOffsets);
  } else {
    VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
        encodeColumnTyped,
        typeKind,
        decodedVector,
        numRows,
        descending,
        nullLast,
        rowOffsets);
  }
}

RowVectorPtr KeyEncoder::createIncrementedBounds(
    const RowVectorPtr& bounds) const {
  const auto& children = bounds->children();
  const auto numRows = bounds->size();
  const auto numKeyColumns = keyChannels_.size();
  VELOX_CHECK_EQ(children.size(), numKeyColumns);

  // Create result vectors for all rows.
  std::vector<VectorPtr> newChildren;
  newChildren.reserve(numKeyColumns);
  for (const auto& child : children) {
    newChildren.push_back(BaseVector::create(child->type(), numRows, pool_));
  }

  // Copy all values from the source rows.
  for (size_t i = 0; i < numKeyColumns; ++i) {
    newChildren[i]->copy(children[i].get(), 0, 0, numRows);
  }

  // Process each row independently.
  for (vector_size_t row = 0; row < numRows; ++row) {
    // Increment from rightmost key column (least significant) to leftmost.
    // Stop when we find a column that can be incremented.
    bool succeeded = false;
    for (int col = numKeyColumns - 1; col >= 0; --col) {
      const bool descending = !sortOrders_[col].isAscending();
      const bool nullLast = !sortOrders_[col].isNullsFirst();
      if (incrementColumnValue(
              children[col], row, descending, nullLast, newChildren[col])) {
        succeeded = true;
        break;
      }
    }
    if (!succeeded) {
      return nullptr;
    }
  }

  return std::make_shared<RowVector>(
      pool_, bounds->type(), nullptr, numRows, std::move(newChildren));
}

uint64_t KeyEncoder::estimateEncodedSize() {
  const auto numRows = decodedVector_.size();
  encodedSizes_.resize(numRows, 0);

  // Process each key channel in columnar order.
  for (auto i = 0; i < keyChannels_.size(); ++i) {
    ScratchPtr<vector_size_t, 128> decodedRowsHolder(scratch_);
    auto* decodedRows = decodedRowsHolder.get(numRows);
    for (auto row = 0; row < numRows; ++row) {
      decodedRows[row] = decodedVector_.index(row);
    }

    estimateEncodedColumnSize(
        childDecodedVectors_[i],
        folly::Range<const vector_size_t*>(decodedRows, numRows),
        encodedSizes_,
        scratch_);
  }
  return std::accumulate(encodedSizes_.begin(), encodedSizes_.end(), 0);
}

std::vector<std::string> KeyEncoder::encode(const RowVectorPtr& input) {
  std::vector<char> buffer;
  std::vector<std::string_view> encodedKeys;
  // Call the templated encode method with a buffer allocator lambda
  encode(input, encodedKeys, [&buffer](size_t size) -> void* {
    buffer.resize(size);
    return buffer.data();
  });

  // Copy encoded keys to result.
  std::vector<std::string> result;
  result.reserve(encodedKeys.size());
  for (const auto& key : encodedKeys) {
    result.emplace_back(key.data(), key.size());
  }
  return result;
}

std::vector<EncodedKeyBounds> KeyEncoder::encodeIndexBounds(
    const IndexBounds& indexBounds) {
  VELOX_CHECK(indexBounds.validate());

  const auto numRows = indexBounds.numRows();
  std::vector<EncodedKeyBounds> results(numRows);

  // Encode lower bounds if present.
  if (indexBounds.lowerBound.has_value()) {
    const auto& lowerBound = indexBounds.lowerBound.value();
    RowVectorPtr lowerBoundToEncode = lowerBound.bound;

    // For exclusive lower bound, bump up all rows before encoding.
    if (!lowerBound.inclusive) {
      lowerBoundToEncode = createIncrementedBounds(lowerBoundToEncode);
      VELOX_CHECK_NOT_NULL(
          lowerBoundToEncode,
          "Failed to bump up lower bound for exclusive range");
    }

    auto encodedKeys = encode(lowerBoundToEncode);
    VELOX_CHECK_EQ(encodedKeys.size(), numRows);
    for (vector_size_t row = 0; row < numRows; ++row) {
      results[row].lowerKey = std::move(encodedKeys[row]);
    }
  }

  // Encode upper bounds if present.
  if (indexBounds.upperBound.has_value()) {
    const auto& upperBound = indexBounds.upperBound.value();
    RowVectorPtr upperBoundToEncode = upperBound.bound;

    // For inclusive upper bound, bump up all rows before encoding.
    if (upperBound.inclusive) {
      upperBoundToEncode = createIncrementedBounds(upperBound.bound);
      // If bump up failed, upperKey remains std::nullopt (unbounded).
    }

    if (upperBoundToEncode != nullptr) {
      auto encodedKeys = encode(upperBoundToEncode);
      VELOX_CHECK_EQ(encodedKeys.size(), numRows);
      for (vector_size_t row = 0; row < numRows; ++row) {
        results[row].upperKey = std::move(encodedKeys[row]);
      }
    }
  }

  return results;
}

} // namespace facebook::velox::serializer
