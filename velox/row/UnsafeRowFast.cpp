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
#include "velox/row/UnsafeRowFast.h"
#include "velox/common/base/RawVector.h"

namespace facebook::velox::row {

namespace {
static const int32_t kFieldWidth = 8;

int32_t alignBits(int32_t numBits) {
  return bits::nwords(numBits) * 8;
}

int32_t alignBytes(int32_t numBytes) {
  return bits::roundUp(numBytes, 8);
}

bool isFixedWidth(const TypePtr& type) {
  return type->isFixedWidth() && !type->isLongDecimal();
}

FOLLY_ALWAYS_INLINE void writeFixedWidth(
    char* buffer,
    const char* rawData,
    vector_size_t index,
    size_t valueBytes) {
  memcpy(buffer, rawData + index * valueBytes, valueBytes);
}

FOLLY_ALWAYS_INLINE void writeTimestamp(
    char* buffer,
    const Timestamp& timestamp) {
  // Write micros(int64_t) for timestamp value.
  const auto micros = timestamp.toMicros();
  memcpy(buffer, &micros, sizeof(int64_t));
}

FOLLY_ALWAYS_INLINE void writeString(
    char* buffer,
    char* rowBase,
    size_t& variableWidthOffset,
    const StringView& value) {
  uint64_t sizeAndOffset = variableWidthOffset << 32 | value.size();
  *reinterpret_cast<uint64_t*>(buffer) = sizeAndOffset;

  if (!value.empty()) {
    memcpy(rowBase + variableWidthOffset, value.data(), value.size());
    variableWidthOffset += alignBytes(value.size());
  }
}

FOLLY_ALWAYS_INLINE void writeLongDecimal(
    char* buffer,
    char* rowBase,
    size_t& variableWidthOffset,
    const int128_t& value) {
  auto serializedLength =
      DecimalUtil::toByteArray(value, rowBase + variableWidthOffset);
  uint64_t sizeAndOffset = variableWidthOffset << 32 | serializedLength;
  *reinterpret_cast<uint64_t*>(buffer) = sizeAndOffset;
  variableWidthOffset += alignBytes(serializedLength);
}

// Serialize a child vector of a row type within a list of rows.
// Write the serialized data at offsets of buffer row by row.
// Update offsets with the actual serialized size.
template <TypeKind kind>
void serializeTyped(
    const raw_vector<vector_size_t>& rows,
    uint32_t childIdx,
    DecodedVector& decoded,
    size_t valueBytes,
    const raw_vector<char*>& nulls,
    const raw_vector<char*>& data,
    std::vector<size_t>& /*unused*/) {
  const auto* rawData = decoded.data<char>();
  const auto childOffset = childIdx * kFieldWidth;
  if (!decoded.mayHaveNulls()) {
    for (auto i = 0; i < rows.size(); ++i) {
      writeFixedWidth(
          data[i] + childOffset, rawData, decoded.index(rows[i]), valueBytes);
    }
  } else {
    for (auto i = 0; i < rows.size(); ++i) {
      if (decoded.isNullAt(rows[i])) {
        bits::setBit(nulls[i], childIdx, true);
      } else {
        writeFixedWidth(
            data[i] + childOffset, rawData, decoded.index(rows[i]), valueBytes);
      }
    }
  }
}

template <>
void serializeTyped<TypeKind::UNKNOWN>(
    const raw_vector<vector_size_t>& rows,
    uint32_t childIdx,
    DecodedVector& /* unused */,
    size_t /* unused */,
    const raw_vector<char*>& nulls,
    const raw_vector<char*>& /*unused*/,
    std::vector<size_t>& /*unused*/) {
  for (auto i = 0; i < rows.size(); ++i) {
    bits::setBit(nulls[i], childIdx, true);
  }
}

template <>
void serializeTyped<TypeKind::BOOLEAN>(
    const raw_vector<vector_size_t>& rows,
    uint32_t childIdx,
    DecodedVector& decoded,
    size_t /* unused */,
    const raw_vector<char*>& nulls,
    const raw_vector<char*>& data,
    std::vector<size_t>& /*unused*/) {
  const auto childOffset = childIdx * kFieldWidth;
  if (!decoded.mayHaveNulls()) {
    for (auto i = 0; i < rows.size(); ++i) {
      *reinterpret_cast<bool*>(data[i] + childOffset) =
          decoded.valueAt<bool>(rows[i]);
    }
  } else {
    for (auto i = 0; i < rows.size(); ++i) {
      if (decoded.isNullAt(rows[i])) {
        bits::setBit(nulls[i], childIdx, true);
      } else {
        // Write 1 byte for bool type.
        *reinterpret_cast<bool*>(data[i] + childOffset) =
            decoded.valueAt<bool>(rows[i]);
      }
    }
  }
}

template <>
void serializeTyped<TypeKind::TIMESTAMP>(
    const raw_vector<vector_size_t>& rows,
    uint32_t childIdx,
    DecodedVector& decoded,
    size_t /* unused */,
    const raw_vector<char*>& nulls,
    const raw_vector<char*>& data,
    std::vector<size_t>& /*unused*/) {
  const auto childOffset = childIdx * kFieldWidth;
  const auto* rawData = decoded.data<Timestamp>();
  if (!decoded.mayHaveNulls()) {
    for (auto i = 0; i < rows.size(); ++i) {
      auto index = decoded.index(rows[i]);
      writeTimestamp(data[i] + childOffset, rawData[index]);
    }
  } else {
    for (auto i = 0; i < rows.size(); ++i) {
      if (decoded.isNullAt(rows[i])) {
        bits::setBit(nulls[i], childIdx, true);
      } else {
        auto index = decoded.index(rows[i]);
        writeTimestamp(data[i] + childOffset, rawData[index]);
      }
    }
  }
}

template <>
void serializeTyped<TypeKind::VARCHAR>(
    const raw_vector<vector_size_t>& rows,
    uint32_t childIdx,
    DecodedVector& decoded,
    size_t /*unused*/,
    const raw_vector<char*>& nulls,
    const raw_vector<char*>& data,
    std::vector<size_t>& variableWidthOffsets) {
  const auto childOffset = childIdx * kFieldWidth;
  if (!decoded.mayHaveNulls()) {
    for (auto i = 0; i < rows.size(); ++i) {
      writeString(
          data[i] + childOffset,
          nulls[i],
          variableWidthOffsets[i],
          decoded.valueAt<StringView>(rows[i]));
    }
  } else {
    for (auto i = 0; i < rows.size(); ++i) {
      if (decoded.isNullAt(rows[i])) {
        bits::setBit(nulls[i], childIdx, true);
      } else {
        writeString(
            data[i] + childOffset,
            nulls[i],
            variableWidthOffsets[i],
            decoded.valueAt<StringView>(rows[i]));
      }
    }
  }
}

template <>
void serializeTyped<TypeKind::VARBINARY>(
    const raw_vector<vector_size_t>& rows,
    uint32_t childIdx,
    DecodedVector& decoded,
    size_t valueBytes,
    const raw_vector<char*>& nulls,
    const raw_vector<char*>& data,
    std::vector<size_t>& variableWidthOffsets) {
  serializeTyped<TypeKind::VARCHAR>(
      rows, childIdx, decoded, valueBytes, nulls, data, variableWidthOffsets);
}

template <>
void serializeTyped<TypeKind::HUGEINT>(
    const raw_vector<vector_size_t>& rows,
    uint32_t childIdx,
    DecodedVector& decoded,
    size_t /*unused*/,
    const raw_vector<char*>& nulls,
    const raw_vector<char*>& data,
    std::vector<size_t>& variableWidthOffsets) {
  const auto childOffset = childIdx * kFieldWidth;
  if (!decoded.mayHaveNulls()) {
    for (auto i = 0; i < rows.size(); ++i) {
      writeLongDecimal(
          data[i] + childOffset,
          nulls[i],
          variableWidthOffsets[i],
          decoded.valueAt<int128_t>(rows[i]));
    }
  } else {
    for (auto i = 0; i < rows.size(); ++i) {
      if (decoded.isNullAt(rows[i])) {
        bits::setBit(nulls[i], childIdx, true);
      } else {
        writeLongDecimal(
            data[i] + childOffset,
            nulls[i],
            variableWidthOffsets[i],
            decoded.valueAt<int128_t>(rows[i]));
      }
    }
  }
}
} // namespace

// static
std::optional<int32_t> UnsafeRowFast::fixedRowSize(const RowTypePtr& rowType) {
  for (const auto& child : rowType->children()) {
    if (!isFixedWidth(child)) {
      return std::nullopt;
    }
  }

  const size_t numFields = rowType->size();
  const size_t nullLength = alignBits(numFields);

  return nullLength + numFields * kFieldWidth;
}

UnsafeRowFast::UnsafeRowFast(const RowVectorPtr& vector)
    : typeKind_{vector->typeKind()}, decoded_{*vector} {
  initialize(vector->type());
}

UnsafeRowFast::UnsafeRowFast(const VectorPtr& vector)
    : typeKind_{vector->typeKind()}, decoded_{*vector} {
  initialize(vector->type());
}

void UnsafeRowFast::initialize(const TypePtr& type) {
  auto base = decoded_.base();
  switch (typeKind_) {
    case TypeKind::ARRAY: {
      auto arrayBase = base->as<ArrayVector>();
      children_.push_back(UnsafeRowFast(arrayBase->elements()));
      childIsFixedWidth_.push_back(isFixedWidth(arrayBase->elements()->type()));
      break;
    }
    case TypeKind::MAP: {
      auto mapBase = base->as<MapVector>();
      children_.push_back(UnsafeRowFast(mapBase->mapKeys()));
      children_.push_back(UnsafeRowFast(mapBase->mapValues()));
      childIsFixedWidth_.push_back(isFixedWidth(mapBase->mapKeys()->type()));
      childIsFixedWidth_.push_back(isFixedWidth(mapBase->mapValues()->type()));
      break;
    }
    case TypeKind::ROW: {
      auto rowBase = base->as<RowVector>();
      for (const auto& child : rowBase->children()) {
        children_.push_back(UnsafeRowFast(child));
        const auto childIsFixedWidth = isFixedWidth(child->type());
        childIsFixedWidth_.push_back(childIsFixedWidth);
        if (!childIsFixedWidth) {
          hasVariableWidth_ = true;
        }
      }

      rowNullBytes_ = alignBits(type->size());
      break;
    }
    case TypeKind::BOOLEAN:
      valueBytes_ = 1;
      fixedWidthTypeKind_ = true;
      break;
    case TypeKind::TINYINT:
      [[fallthrough]];
    case TypeKind::SMALLINT:
      [[fallthrough]];
    case TypeKind::INTEGER:
      [[fallthrough]];
    case TypeKind::BIGINT:
      [[fallthrough]];
    case TypeKind::REAL:
      [[fallthrough]];
    case TypeKind::DOUBLE:
      [[fallthrough]];
    case TypeKind::UNKNOWN:
      valueBytes_ = type->cppSizeInBytes();
      fixedWidthTypeKind_ = true;
      supportsBulkCopy_ = decoded_.isIdentityMapping();
      break;
    case TypeKind::TIMESTAMP:
      valueBytes_ = sizeof(int64_t);
      fixedWidthTypeKind_ = true;
      break;
    case TypeKind::HUGEINT:
      [[fallthrough]];
    case TypeKind::VARCHAR:
      [[fallthrough]];
    case TypeKind::VARBINARY:
      // Nothing to do.
      break;
    default:
      VELOX_UNSUPPORTED("Unsupported type: {}", type->toString());
  }
}

int32_t UnsafeRowFast::rowSize(vector_size_t index) {
  return rowRowSize(index);
}

int32_t UnsafeRowFast::variableWidthRowSize(vector_size_t index) {
  switch (typeKind_) {
    case TypeKind::VARCHAR:
      [[fallthrough]];
    case TypeKind::VARBINARY: {
      auto value = decoded_.valueAt<StringView>(index);
      return alignBytes(value.size());
    }
    case TypeKind::HUGEINT:
      return DecimalUtil::getByteArrayLength(decoded_.valueAt<int128_t>(index));
    case TypeKind::ARRAY:
      return arrayRowSize(index);
    case TypeKind::MAP:
      return mapRowSize(index);
    case TypeKind::ROW:
      return rowRowSize(index);
    default:
      VELOX_UNREACHABLE(
          "Unexpected type kind: {}", mapTypeKindToName(typeKind_));
  };
}

bool UnsafeRowFast::isNullAt(vector_size_t index) {
  return decoded_.isNullAt(index);
}

int32_t UnsafeRowFast::serialize(vector_size_t index, char* buffer) {
  return serializeRow(index, buffer);
}

void UnsafeRowFast::serialize(
    vector_size_t offset,
    vector_size_t size,
    char* buffer,
    const size_t* bufferOffsets) {
  if (size == 1) {
    (void)serializeRow(offset, buffer + *bufferOffsets);
    return;
  }
  return serializeRow(offset, size, buffer, bufferOffsets);
}

void UnsafeRowFast::serializeFixedWidth(vector_size_t index, char* buffer) {
  VELOX_DCHECK(fixedWidthTypeKind_);
  switch (typeKind_) {
    case TypeKind::BOOLEAN:
      *reinterpret_cast<bool*>(buffer) = decoded_.valueAt<bool>(index);
      break;
    case TypeKind::TIMESTAMP:
      *reinterpret_cast<int64_t*>(buffer) =
          decoded_.valueAt<Timestamp>(index).toMicros();
      break;
    default:
      memcpy(
          buffer,
          decoded_.data<char>() + decoded_.index(index) * valueBytes_,
          valueBytes_);
  }
}

void UnsafeRowFast::serializeFixedWidth(
    vector_size_t offset,
    vector_size_t size,
    char* buffer) {
  VELOX_DCHECK(supportsBulkCopy_);
  // decoded_.data<char>() can be null if all values are null.
  if (decoded_.data<char>()) {
    memcpy(
        buffer,
        decoded_.data<char>() + decoded_.index(offset) * valueBytes_,
        valueBytes_ * size);
  }
}

int32_t UnsafeRowFast::serializeVariableWidth(
    vector_size_t index,
    char* buffer) {
  switch (typeKind_) {
    case TypeKind::VARCHAR:
      [[fallthrough]];
    case TypeKind::VARBINARY: {
      auto value = decoded_.valueAt<StringView>(index);
      memcpy(buffer, value.data(), value.size());
      return value.size();
    }
    case TypeKind::HUGEINT: {
      auto value = decoded_.valueAt<int128_t>(index);
      return DecimalUtil::toByteArray(value, buffer);
    }
    case TypeKind::ARRAY:
      return serializeArray(index, buffer);
    case TypeKind::MAP:
      return serializeMap(index, buffer);
    case TypeKind::ROW:
      return serializeRow(index, buffer);
    default:
      VELOX_UNREACHABLE(
          "Unexpected type kind: {}", mapTypeKindToName(typeKind_));
  };
}

int32_t UnsafeRowFast::arrayRowSize(vector_size_t index) {
  auto baseIndex = decoded_.index(index);

  // array size | null bits | fixed-width data | variable-width data
  auto arrayBase = decoded_.base()->asUnchecked<ArrayVector>();
  auto offset = arrayBase->offsetAt(baseIndex);
  auto size = arrayBase->sizeAt(baseIndex);

  return arrayRowSize(children_[0], offset, size, childIsFixedWidth_[0]);
}

int32_t UnsafeRowFast::serializeArray(vector_size_t index, char* buffer) {
  auto baseIndex = decoded_.index(index);

  // array size | null bits | fixed-width data | variable-width data
  auto arrayBase = decoded_.base()->asUnchecked<ArrayVector>();
  auto offset = arrayBase->offsetAt(baseIndex);
  auto size = arrayBase->sizeAt(baseIndex);

  return serializeAsArray(
      children_[0], offset, size, childIsFixedWidth_[0], buffer);
}

int32_t UnsafeRowFast::mapRowSize(vector_size_t index) {
  auto baseIndex = decoded_.index(index);

  //  size of serialized keys array in bytes | <keys array> | <values array>

  auto mapBase = decoded_.base()->asUnchecked<MapVector>();
  auto offset = mapBase->offsetAt(baseIndex);
  auto size = mapBase->sizeAt(baseIndex);

  return kFieldWidth +
      arrayRowSize(children_[0], offset, size, childIsFixedWidth_[0]) +
      arrayRowSize(children_[1], offset, size, childIsFixedWidth_[1]);
}

int32_t UnsafeRowFast::serializeMap(vector_size_t index, char* buffer) {
  auto baseIndex = decoded_.index(index);

  //  size of serialized keys array in bytes | <keys array> | <values array>

  auto mapBase = decoded_.base()->asUnchecked<MapVector>();
  auto offset = mapBase->offsetAt(baseIndex);
  auto size = mapBase->sizeAt(baseIndex);

  int32_t serializedBytes = kFieldWidth;

  auto keysSerializedBytes = serializeAsArray(
      children_[0],
      offset,
      size,
      childIsFixedWidth_[0],
      buffer + serializedBytes);
  serializedBytes += keysSerializedBytes;

  auto valuesSerializedBytes = serializeAsArray(
      children_[1],
      offset,
      size,
      childIsFixedWidth_[1],
      buffer + serializedBytes);
  serializedBytes += valuesSerializedBytes;

  // Write the size of serialized keys.
  *reinterpret_cast<int64_t*>(buffer) = keysSerializedBytes;

  return serializedBytes;
}

int32_t UnsafeRowFast::arrayRowSize(
    UnsafeRowFast& elements,
    vector_size_t offset,
    vector_size_t size,
    bool fixedWidth) {
  int32_t nullBytes = alignBits(size);

  int32_t rowSize = kFieldWidth + nullBytes;
  if (fixedWidth) {
    return rowSize + size * elements.valueBytes();
  }

  rowSize += size * kFieldWidth;

  for (auto i = 0; i < size; ++i) {
    if (!elements.isNullAt(offset + i)) {
      rowSize += alignBytes(elements.variableWidthRowSize(offset + i));
    }
  }

  return rowSize;
}

int32_t UnsafeRowFast::serializeAsArray(
    UnsafeRowFast& elements,
    vector_size_t offset,
    vector_size_t size,
    bool fixedWidth,
    char* buffer) {
  // array size | null bits | fixed-width data | variable-width data

  // Write array size.
  *reinterpret_cast<int64_t*>(buffer) = size;

  int32_t nullBytes = alignBits(size);

  int32_t nullsOffset = sizeof(int64_t);
  int32_t fixedWidthOffset = nullsOffset + nullBytes;

  auto childSize = fixedWidth ? elements.valueBytes() : kFieldWidth;

  int64_t variableWidthOffset = fixedWidthOffset + size * childSize;

  if (elements.supportsBulkCopy_) {
    if (elements.decoded_.mayHaveNulls()) {
      for (auto i = 0; i < size; ++i) {
        if (elements.isNullAt(offset + i)) {
          bits::setBit(buffer + nullsOffset, i, true);
        }
      }
    }
    elements.serializeFixedWidth(offset, size, buffer + fixedWidthOffset);
    return variableWidthOffset;
  }

  for (auto i = 0; i < size; ++i) {
    if (elements.isNullAt(offset + i)) {
      bits::setBit(buffer + nullsOffset, i, true);
    } else {
      if (fixedWidth) {
        elements.serializeFixedWidth(
            offset + i, buffer + fixedWidthOffset + i * childSize);
      } else {
        auto serializedBytes = elements.serializeVariableWidth(
            offset + i, buffer + variableWidthOffset);

        // Write size and offset.
        uint64_t sizeAndOffset = variableWidthOffset << 32 | serializedBytes;
        reinterpret_cast<uint64_t*>(buffer + fixedWidthOffset)[i] =
            sizeAndOffset;

        variableWidthOffset += alignBytes(serializedBytes);
      }
    }
  }
  return variableWidthOffset;
}

int32_t UnsafeRowFast::rowRowSize(vector_size_t index) {
  auto childIndex = decoded_.index(index);

  const auto numFields = children_.size();
  int32_t size = rowNullBytes_ + numFields * kFieldWidth;

  for (auto i = 0; i < numFields; ++i) {
    if (!childIsFixedWidth_[i] && !children_[i].isNullAt(childIndex)) {
      size += alignBytes(children_[i].variableWidthRowSize(childIndex));
    }
  }

  return size;
}

int32_t UnsafeRowFast::serializeRow(vector_size_t index, char* buffer) {
  auto childIndex = decoded_.index(index);

  int64_t variableWidthOffset = rowNullBytes_ + kFieldWidth * children_.size();

  for (auto i = 0; i < children_.size(); ++i) {
    auto& child = children_[i];

    // Write null bit.
    if (child.isNullAt(childIndex)) {
      bits::setBit(buffer, i, true);
      continue;
    }

    // Write value.
    if (childIsFixedWidth_[i]) {
      child.serializeFixedWidth(
          childIndex, buffer + rowNullBytes_ + i * kFieldWidth);
    } else {
      auto size = child.serializeVariableWidth(
          childIndex, buffer + variableWidthOffset);
      // Write size and offset.
      uint64_t sizeAndOffset = variableWidthOffset << 32 | size;
      reinterpret_cast<uint64_t*>(buffer + rowNullBytes_)[i] = sizeAndOffset;

      variableWidthOffset += alignBytes(size);
    }
  }

  return variableWidthOffset;
}

void UnsafeRowFast::serializeRow(
    vector_size_t offset,
    vector_size_t size,
    char* buffer,
    const size_t* bufferOffsets) {
  raw_vector<vector_size_t> rows(size);
  raw_vector<char*> nulls(size);
  raw_vector<char*> data(size);
  if (decoded_.isIdentityMapping()) {
    std::iota(rows.begin(), rows.end(), offset);
  } else {
    for (auto i = 0; i < size; ++i) {
      rows[i] = decoded_.index(offset + i);
    }
  }

  // After serializing variable-width column, the 'variableWidthOffsets' are
  // updated accordingly.
  std::vector<size_t> variableWidthOffsets;
  if (hasVariableWidth_) {
    variableWidthOffsets.resize(size);
  }

  const size_t fixedFieldLength = kFieldWidth * children_.size();
  for (auto i = 0; i < size; ++i) {
    nulls[i] = buffer + bufferOffsets[i];
    data[i] = buffer + bufferOffsets[i] + rowNullBytes_;
    if (hasVariableWidth_) {
      variableWidthOffsets[i] = rowNullBytes_ + fixedFieldLength;
    }
  }

  // Fixed-width and varchar/varbinary types are serialized using the vectorized
  // API 'serializedTyped'. Other data types are serialized row-by-row.
  for (auto childIdx = 0; childIdx < children_.size(); ++childIdx) {
    auto& child = children_[childIdx];
    if (childIsFixedWidth_[childIdx] || child.typeKind_ == TypeKind::HUGEINT ||
        child.typeKind_ == TypeKind::VARBINARY ||
        child.typeKind_ == TypeKind::VARCHAR) {
      VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH_ALL(
          serializeTyped,
          child.typeKind_,
          rows,
          childIdx,
          child.decoded_,
          child.valueBytes_,
          nulls,
          data,
          variableWidthOffsets);
    } else {
      const auto mayHaveNulls = child.decoded_.mayHaveNulls();
      const auto childOffset = childIdx * kFieldWidth;
      for (auto i = 0; i < rows.size(); ++i) {
        if (mayHaveNulls && child.isNullAt(rows[i])) {
          bits::setBit(nulls[i], childIdx, true);
        } else {
          // Write non-null variable-width value.
          auto size = child.serializeVariableWidth(
              rows[i], nulls[i] + variableWidthOffsets[i]);
          // Write size and offset.
          uint64_t sizeAndOffset = variableWidthOffsets[i] << 32 | size;
          *reinterpret_cast<uint64_t*>(data[i] + childOffset) = sizeAndOffset;

          variableWidthOffsets[i] += alignBytes(size);
        }
      }
    }
  }
}

} // namespace facebook::velox::row
