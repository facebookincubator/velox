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
#include "velox/serializers/KeyDecoder.h"

#include <cstring>
#include <limits>
#include <type_traits>

#include <folly/lang/Bits.h>

#include "velox/common/base/Exceptions.h"
#include "velox/type/Timestamp.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::serializer {
namespace {

// Validates if a type is a valid index column type.
// Only primitive scalar types are supported except UNKNOWN and HUGEINT.
bool isValidIndexColumnType(const TypePtr& type) {
  return type->isPrimitiveType() && type->kind() != TypeKind::UNKNOWN &&
      type->kind() != TypeKind::HUGEINT;
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

RowTypePtr makeOutputType(
    const RowTypePtr& inputType,
    const std::vector<std::string>& keyColumns) {
  const auto keyChannels = getKeyChannels(inputType, keyColumns);
  std::vector<TypePtr> childTypes;
  childTypes.reserve(keyChannels.size());
  for (const auto channel : keyChannels) {
    childTypes.push_back(inputType->childAt(channel));
  }
  return ROW(keyColumns, std::move(childTypes));
}

vector_size_t checkedVectorSize(size_t size) {
  VELOX_CHECK_LE(
      size,
      static_cast<size_t>(std::numeric_limits<vector_size_t>::max()),
      "Row count exceeds vector_size_t maximum: {}",
      size);
  return static_cast<vector_size_t>(size);
}

column_index_t checkedColumnIndex(size_t index) {
  VELOX_CHECK_LE(
      index,
      static_cast<size_t>(std::numeric_limits<column_index_t>::max()),
      "Column index exceeds column_index_t maximum: {}",
      index);
  return static_cast<column_index_t>(index);
}

StringView makeStringView(std::string_view value) {
  VELOX_CHECK_LE(
      value.size(),
      static_cast<size_t>(std::numeric_limits<int32_t>::max()),
      "Decoded string exceeds StringView maximum length: {}",
      value.size());
  return StringView(value.data(), static_cast<int32_t>(value.size()));
}

struct Cursor {
  const std::string_view encodedKey;
  size_t offset_{0};

  explicit Cursor(std::string_view encodedKey) : encodedKey{encodedKey} {}

  uint8_t readRawByte() {
    VELOX_CHECK_LT(
        offset_,
        encodedKey.size(),
        "Malformed encoded key: unexpected end of key at offset {}",
        offset());
    return static_cast<uint8_t>(encodedKey.at(offset_++));
  }

  template <typename T>
  T readRawBytes() {
    VELOX_CHECK_GE(
        remaining(),
        sizeof(T),
        "Malformed encoded key: need {} byte(s) at offset {}, only {} byte(s) remain",
        sizeof(T),
        offset(),
        remaining());

    T value;
    auto* bytes = reinterpret_cast<uint8_t*>(&value);
    for (size_t i = 0; i < sizeof(T); ++i) {
      bytes[i] = readRawByte();
    }
    return value;
  }

  size_t offset() const {
    return offset_;
  }

  size_t remaining() const {
    return encodedKey.size() - offset_;
  }
};

// Reverses KeyEncoder's bytewise inversion for descending sort order.
FOLLY_ALWAYS_INLINE uint8_t decodeByte(uint8_t value, bool descending) {
  return descending ? static_cast<uint8_t>(0xff ^ value) : value;
}

bool decodeNullMarker(Cursor& cursor, bool nullLast) {
  const auto marker = cursor.readRawByte();
  const auto nullByte = static_cast<uint8_t>(nullLast);
  const auto nonNullByte = static_cast<uint8_t>(!nullLast);
  VELOX_CHECK(
      marker == nullByte || marker == nonNullByte,
      "Malformed encoded key: invalid null marker {} at offset {}",
      static_cast<uint32_t>(marker),
      cursor.offset() - 1);
  return marker == nullByte;
}

template <typename T>
T decodeFixedWidth(Cursor& cursor, bool descending) {
  using UnsignedT = std::make_unsigned_t<T>;

  UnsignedT value = cursor.readRawBytes<UnsignedT>();
  if (descending) {
    value = ~value;
  }
  value = folly::Endian::big(value);

  if constexpr (std::is_signed_v<T>) {
    constexpr int kSignBitShift = sizeof(T) * 8 - 1;
    value ^= (UnsignedT{1} << kSignBitShift);
  }

  return static_cast<T>(value);
}

bool decodeBoolean(Cursor& cursor, bool descending) {
  const auto decoded = decodeByte(cursor.readRawByte(), descending);
  VELOX_CHECK(
      decoded == 1 || decoded == 2,
      "Malformed encoded key: invalid boolean byte {} at offset {}",
      static_cast<uint32_t>(decoded),
      cursor.offset() - 1);
  return decoded == 2;
}

std::string decodeString(Cursor& cursor, bool descending) {
  std::string result;

  while (true) {
    const auto decoded = decodeByte(cursor.readRawByte(), descending);
    if (decoded == 0) {
      return result;
    }

    if (decoded == 1) {
      const auto decodedEscaped = decodeByte(cursor.readRawByte(), descending);
      VELOX_CHECK(
          decodedEscaped == 0 || decodedEscaped == 1,
          "Malformed encoded key: invalid escaped string byte {} at offset {}",
          static_cast<uint32_t>(decodedEscaped),
          cursor.offset() - 1);
      result.push_back(static_cast<char>(decodedEscaped));
      continue;
    }

    result.push_back(static_cast<char>(decoded));
  }
}

double decodeDouble(Cursor& cursor, bool descending) {
  constexpr uint64_t kSignBit = 1ULL << 63;

  const uint64_t orderedValue = decodeFixedWidth<uint64_t>(cursor, descending);
  const uint64_t bits = (orderedValue & kSignBit) != 0
      ? (orderedValue ^ kSignBit)
      : ~orderedValue;

  double result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

float decodeReal(Cursor& cursor, bool descending) {
  constexpr uint32_t kSignBit = 1U << 31;

  const uint32_t orderedValue = decodeFixedWidth<uint32_t>(cursor, descending);
  const uint32_t bits = (orderedValue & kSignBit) != 0
      ? (orderedValue ^ kSignBit)
      : ~orderedValue;

  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

void decodeValue(
    Cursor& cursor,
    const TypePtr& type,
    const core::SortOrder& sortOrder,
    vector_size_t row,
    VectorPtr& output) {
  const bool nullLast = !sortOrder.isNullsFirst();
  const bool descending = !sortOrder.isAscending();

  if (decodeNullMarker(cursor, nullLast)) {
    output->setNull(row, true);
    return;
  }

  if (type->isDate()) {
    output->asFlatVector<int32_t>()->set(
        row, decodeFixedWidth<int32_t>(cursor, descending));
    return;
  }

  switch (type->kind()) {
    case TypeKind::BIGINT:
      output->asFlatVector<int64_t>()->set(
          row, decodeFixedWidth<int64_t>(cursor, descending));
      return;
    case TypeKind::INTEGER:
      output->asFlatVector<int32_t>()->set(
          row, decodeFixedWidth<int32_t>(cursor, descending));
      return;
    case TypeKind::SMALLINT:
      output->asFlatVector<int16_t>()->set(
          row, decodeFixedWidth<int16_t>(cursor, descending));
      return;
    case TypeKind::TINYINT:
      output->asFlatVector<int8_t>()->set(
          row, decodeFixedWidth<int8_t>(cursor, descending));
      return;
    case TypeKind::DOUBLE:
      output->asFlatVector<double>()->set(
          row, decodeDouble(cursor, descending));
      return;
    case TypeKind::REAL:
      output->asFlatVector<float>()->set(row, decodeReal(cursor, descending));
      return;
    case TypeKind::BOOLEAN:
      output->asFlatVector<bool>()->set(row, decodeBoolean(cursor, descending));
      return;
    case TypeKind::VARCHAR:
    case TypeKind::VARBINARY: {
      const auto value = decodeString(cursor, descending);
      output->asFlatVector<StringView>()->set(row, makeStringView(value));
      return;
    }
    case TypeKind::TIMESTAMP: {
      const auto seconds = decodeFixedWidth<int64_t>(cursor, descending);
      const auto nanos = decodeFixedWidth<uint64_t>(cursor, descending);
      output->asFlatVector<Timestamp>()->set(row, Timestamp(seconds, nanos));
      return;
    }
    case TypeKind::HUGEINT:
    case TypeKind::ARRAY:
    case TypeKind::MAP:
    case TypeKind::ROW:
    case TypeKind::UNKNOWN:
    case TypeKind::FUNCTION:
    case TypeKind::OPAQUE:
    case TypeKind::INVALID:
      VELOX_UNSUPPORTED("Unsupported type: {}", type->kind());
  }
}

} // namespace

// static.
std::unique_ptr<KeyDecoder> KeyDecoder::create(
    const std::vector<std::string>& keyColumns,
    RowTypePtr inputType,
    std::vector<core::SortOrder> sortOrders,
    memory::MemoryPool* pool) {
  return std::unique_ptr<KeyDecoder>(new KeyDecoder(
      keyColumns, std::move(inputType), std::move(sortOrders), pool));
}

KeyDecoder::KeyDecoder(
    const std::vector<std::string>& keyColumns,
    RowTypePtr inputType,
    std::vector<core::SortOrder> sortOrders,
    memory::MemoryPool* pool)
    : outputType_{makeOutputType(inputType, keyColumns)},
      sortOrders_{std::move(sortOrders)},
      pool_{pool} {
  VELOX_CHECK_GT(outputType_->size(), 0);
  VELOX_CHECK_EQ(
      outputType_->size(),
      sortOrders_.size(),
      "Size mismatch between key columns and sort orders");

  for (size_t i = 0; i < outputType_->size(); ++i) {
    const auto column = checkedColumnIndex(i);
    const auto& columnType = outputType_->childAt(column);
    VELOX_CHECK(
        isValidIndexColumnType(columnType),
        "Unsupported type for index column '{}': {}",
        outputType_->nameOf(column),
        columnType->toString());
  }
}

RowVectorPtr KeyDecoder::decode(
    std::span<const std::string_view> encodedKeys) const {
  const auto numRows = checkedVectorSize(encodedKeys.size());
  std::vector<VectorPtr> children;
  children.reserve(outputType_->size());
  for (size_t i = 0; i < outputType_->size(); ++i) {
    const auto column = checkedColumnIndex(i);
    children.push_back(
        BaseVector::create(outputType_->childAt(column), numRows, pool_));
  }

  for (vector_size_t row = 0; row < numRows; ++row) {
    Cursor cursor(encodedKeys[static_cast<size_t>(row)]);
    for (size_t column = 0; column < outputType_->size(); ++column) {
      const auto columnIndex = checkedColumnIndex(column);
      decodeValue(
          cursor,
          outputType_->childAt(columnIndex),
          sortOrders_[column],
          row,
          children.at(column));
    }

    VELOX_CHECK_EQ(
        cursor.remaining(),
        0,
        "Malformed encoded key: {} trailing byte(s) remain after decoding row {}",
        cursor.remaining(),
        row);
  }

  return std::make_shared<RowVector>(
      pool_, outputType_, nullptr, numRows, std::move(children));
}

RowVectorPtr KeyDecoder::decode(
    std::span<const std::string> encodedKeys) const {
  std::vector<std::string_view> views;
  views.reserve(encodedKeys.size());
  for (const auto& key : encodedKeys) {
    views.emplace_back(key);
  }
  return decode(std::span<const std::string_view>(views));
}

} // namespace facebook::velox::serializer
