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
#include "velox/row/CompactRow.h"

#include "velox/common/memory/RawVector.h"
#include "velox/vector/FlatMapVector.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::row {
namespace {
constexpr size_t kSizeBytes = sizeof(int32_t);
using TRowSize = uint32_t;

void writeInt32(char* buffer, int32_t n) {
  ::memcpy(buffer, &n, kSizeBytes);
}

int32_t readInt32(const char* buffer) {
  int32_t n;
  ::memcpy(&n, buffer, kSizeBytes);
  return n;
}

FOLLY_ALWAYS_INLINE void writeFixedWidth(
    const char* rawData,
    vector_size_t index,
    size_t valueBytes,
    char* buffer,
    size_t& offset) {
  ::memcpy(buffer + offset, rawData + index * valueBytes, valueBytes);
  offset += valueBytes;
}

FOLLY_ALWAYS_INLINE void
writeTimestamp(const Timestamp& timestamp, char* buffer, size_t& offset) {
  // Write micros(int64_t) for timestamp value.
  const auto timeUs = timestamp.toMicros();
  ::memcpy(buffer + offset, &timeUs, sizeof(int64_t));
  offset += sizeof(int64_t);
}

FOLLY_ALWAYS_INLINE void
writeString(const StringView& value, char* buffer, size_t& offset) {
  writeInt32(buffer + offset, value.size());
  if (!value.empty()) {
    ::memcpy(buffer + offset + kSizeBytes, value.data(), value.size());
  }
  offset += kSizeBytes + value.size();
}

// Serialize the child vector of a row type within a range of consecutive rows.
// Write the serialized data at offsets of buffer row by row.
// Update offsets with the actual serialized size.
template <TypeKind kind>
void serializeTyped(
    const raw_vector<vector_size_t>& rows,
    uint32_t childIdx,
    const DecodedVector& decoded,
    size_t valueBytes,
    const raw_vector<uint8_t*>& nulls,
    char* buffer,
    std::vector<size_t>& offsets) {
  const auto* rawData = decoded.data<char>();
  if (!decoded.mayHaveNulls()) {
    for (auto i = 0; i < rows.size(); ++i) {
      writeFixedWidth(
          rawData, decoded.index(rows[i]), valueBytes, buffer, offsets[i]);
    }
  } else {
    for (auto i = 0; i < rows.size(); ++i) {
      if (decoded.isNullAt(rows[i])) {
        bits::setBit(nulls[i], childIdx, true);
        offsets[i] += valueBytes;
      } else {
        writeFixedWidth(
            rawData, decoded.index(rows[i]), valueBytes, buffer, offsets[i]);
      }
    }
  }
}

template <>
void serializeTyped<TypeKind::UNKNOWN>(
    const raw_vector<vector_size_t>& rows,
    uint32_t childIdx,
    const DecodedVector& /* unused */,
    size_t /* unused */,
    const raw_vector<uint8_t*>& nulls,
    char* /* unused */,
    std::vector<size_t>& /* unused */) {
  for (auto i = 0; i < rows.size(); ++i) {
    bits::setBit(nulls[i], childIdx, true);
  }
}

template <>
void serializeTyped<TypeKind::BOOLEAN>(
    const raw_vector<vector_size_t>& rows,
    uint32_t childIdx,
    const DecodedVector& decoded,
    size_t /* unused */,
    const raw_vector<uint8_t*>& nulls,
    char* buffer,
    std::vector<size_t>& offsets) {
  auto* byte = reinterpret_cast<bool*>(buffer);
  if (!decoded.mayHaveNulls()) {
    for (auto i = 0; i < rows.size(); ++i) {
      byte[offsets[i]++] = decoded.valueAt<bool>(rows[i]);
    }
  } else {
    for (auto i = 0; i < rows.size(); ++i) {
      if (decoded.isNullAt(rows[i])) {
        bits::setBit(nulls[i], childIdx, true);
        offsets[i] += 1;
      } else {
        // Write 1 byte for bool type.
        byte[offsets[i]++] = decoded.valueAt<bool>(rows[i]);
      }
    }
  }
}

template <>
void serializeTyped<TypeKind::TIMESTAMP>(
    const raw_vector<vector_size_t>& rows,
    uint32_t childIdx,
    const DecodedVector& decoded,
    size_t /* unused */,
    const raw_vector<uint8_t*>& nulls,
    char* buffer,
    std::vector<size_t>& offsets) {
  const auto* rawData = decoded.data<Timestamp>();
  if (!decoded.mayHaveNulls()) {
    for (auto i = 0; i < rows.size(); ++i) {
      const auto index = decoded.index(rows[i]);
      writeTimestamp(rawData[index], buffer, offsets[i]);
    }
  } else {
    for (auto i = 0; i < rows.size(); ++i) {
      if (decoded.isNullAt(rows[i])) {
        bits::setBit(nulls[i], childIdx, true);
        offsets[i] += sizeof(int64_t);
      } else {
        const auto index = decoded.index(rows[i]);
        writeTimestamp(rawData[index], buffer, offsets[i]);
      }
    }
  }
}

template <>
void serializeTyped<TypeKind::VARCHAR>(
    const raw_vector<vector_size_t>& rows,
    uint32_t childIdx,
    const DecodedVector& decoded,
    size_t valueBytes,
    const raw_vector<uint8_t*>& nulls,
    char* buffer,
    std::vector<size_t>& offsets) {
  if (!decoded.mayHaveNulls()) {
    for (auto i = 0; i < rows.size(); ++i) {
      writeString(decoded.valueAt<StringView>(rows[i]), buffer, offsets[i]);
    }
  } else {
    for (auto i = 0; i < rows.size(); ++i) {
      if (decoded.isNullAt(rows[i])) {
        bits::setBit(nulls[i], childIdx, true);
      } else {
        writeString(decoded.valueAt<StringView>(rows[i]), buffer, offsets[i]);
      }
    }
  }
}

template <>
void serializeTyped<TypeKind::VARBINARY>(
    const raw_vector<vector_size_t>& rows,
    uint32_t childIdx,
    const DecodedVector& decoded,
    size_t valueBytes,
    const raw_vector<uint8_t*>& nulls,
    char* buffer,
    std::vector<size_t>& offsets) {
  serializeTyped<TypeKind::VARCHAR>(
      rows, childIdx, decoded, valueBytes, nulls, buffer, offsets);
}
} // namespace

CompactRow::CompactRow(const RowVectorPtr& vector)
    : typeKind_{vector->typeKind()}, decoded_{*vector} {
  initialize(vector->type());
}

CompactRow::CompactRow(const VectorPtr& vector)
    : typeKind_{vector->typeKind()}, decoded_{*vector} {
  initialize(vector->type());
}

void CompactRow::initialize(const TypePtr& type) {
  auto base = decoded_.base();
  switch (typeKind_) {
    case TypeKind::ARRAY: {
      auto* arrayBase = base->as<ArrayVector>();
      children_.push_back(CompactRow(arrayBase->elements()));
      childIsFixedWidth_.push_back(
          arrayBase->elements()->type()->isFixedWidth());
      break;
    }
    case TypeKind::MAP: {
      if (base->encoding() == VectorEncoding::Simple::FLAT_MAP) {
        initializeFlatMap();
        break;
      }
      auto* mapBase = base->as<MapVector>();
      VELOX_CHECK_NOT_NULL(
          mapBase,
          "Unsupported map vector encoding for CompactRow: {}",
          base->encoding());
      children_.push_back(CompactRow(mapBase->mapKeys()));
      children_.push_back(CompactRow(mapBase->mapValues()));
      childIsFixedWidth_.push_back(mapBase->mapKeys()->type()->isFixedWidth());
      childIsFixedWidth_.push_back(
          mapBase->mapValues()->type()->isFixedWidth());
      break;
    }
    case TypeKind::ROW: {
      auto* rowBase = base->as<RowVector>();
      for (const auto& child : rowBase->children()) {
        children_.push_back(CompactRow(child));
        childIsFixedWidth_.push_back(child->type()->isFixedWidth());
      }

      rowNullBytes_ = bits::nbytes(type->size());
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
    case TypeKind::HUGEINT:
      [[fallthrough]];
    case TypeKind::REAL:
      [[fallthrough]];
    case TypeKind::DOUBLE:
      valueBytes_ = type->cppSizeInBytes();
      fixedWidthTypeKind_ = true;
      supportsBulkCopy_ = decoded_.isIdentityMapping();
      break;
    case TypeKind::TIMESTAMP:
      valueBytes_ = sizeof(int64_t);
      fixedWidthTypeKind_ = true;
      break;
    case TypeKind::VARCHAR:
      [[fallthrough]];
    case TypeKind::VARBINARY:
      // Nothing to do.
      break;
    case TypeKind::UNKNOWN:
      // UNKNOWN values are always nulls, hence, do not take up space.
      valueBytes_ = 0;
      fixedWidthTypeKind_ = true;
      supportsBulkCopy_ = true;
      break;
    default:
      VELOX_UNSUPPORTED("Unsupported type: {}", type->toString());
  }
}

void CompactRow::initializeFlatMap() {
  const auto* flatMap = decoded_.base()->asUnchecked<FlatMapVector>();
  flatMap_ = true;
  children_.push_back(CompactRow(flatMap->distinctKeys()));
  childIsFixedWidth_.push_back(flatMap->keyType()->isFixedWidth());
  childIsFixedWidth_.push_back(flatMap->valueType()->isFixedWidth());

  const auto size = flatMap->size();
  const auto numKeys = flatMap->numDistinctKeys();
  const auto isRawKind = [](const TypePtr& type) {
    return type->isFixedWidth() && !type->isBoolean() && !type->isTimestamp() &&
        type->kind() != TypeKind::UNKNOWN;
  };
  const auto& valueType = flatMap->valueType();
  const bool rawValues = isRawKind(valueType);
  const bool rawArrays =
      valueType->isArray() && isRawKind(valueType->childAt(0));
  if (rawValues || rawArrays) {
    flatMapRawValues_.assign(numKeys, nullptr);
    flatMapRawNulls_.assign(numKeys, nullptr);
  }
  if (rawArrays) {
    flatMapRawArrays_.assign(numKeys, nullptr);
  }
  flatMapValueChildren_.assign(numKeys, 0);

  // Two passes over the in-map bits: count the entries of each row, then list
  // them.
  flatMapOffsets_.assign(size + 1, 0);
  vector_size_t numEntries{0};
  const auto countEntry = [&](vector_size_t row) {
    ++flatMapOffsets_[row + 1];
    ++numEntries;
  };
  for (column_index_t channel = 0; channel < numKeys; ++channel) {
    numEntries = 0;
    if (const auto* inMap = flatMap->rawInMapsAt(channel)) {
      bits::forEachSetBit(inMap, 0, size, countEntry);
    } else {
      for (vector_size_t row = 0; row < size; ++row) {
        countEntry(row);
      }
    }
    if (numEntries > 0) {
      const auto& values = flatMap->mapValuesAt(channel);
      const auto* array = rawArrays && values != nullptr &&
              values->encoding() == VectorEncoding::Simple::ARRAY
          ? values->asUnchecked<ArrayVector>()
          : nullptr;
      const auto* rawVector = rawValues ? values.get()
          : array != nullptr            ? array->elements().get()
                                        : nullptr;
      if (rawVector != nullptr && rawVector->isFlatEncoding() &&
          rawVector->valuesAsVoid() != nullptr) {
        flatMapRawValues_[channel] =
            static_cast<const char*>(rawVector->valuesAsVoid());
        flatMapRawNulls_[channel] = rawVector->rawNulls();
        if (rawArrays) {
          flatMapRawArrays_[channel] = array;
        }
      } else {
        flatMapValueChildren_[channel] = children_.size();
        children_.push_back(CompactRow(
            values != nullptr ? values
                              : BaseVector::createNullConstant(
                                    valueType, size, flatMap->pool())));
      }
    }
  }
  for (vector_size_t row = 0; row < size; ++row) {
    flatMapOffsets_[row + 1] += flatMapOffsets_[row];
  }
  flatMapChannels_.resize(flatMapOffsets_[size]);
  std::vector<vector_size_t> next(
      flatMapOffsets_.begin(), flatMapOffsets_.end() - 1);
  column_index_t channel{0};
  const auto addEntry = [&](vector_size_t row) {
    flatMapChannels_[next[row]++] = channel;
  };
  for (channel = 0; channel < numKeys; ++channel) {
    if (const auto* inMap = flatMap->rawInMapsAt(channel)) {
      bits::forEachSetBit(inMap, 0, size, addEntry);
    } else {
      for (vector_size_t row = 0; row < size; ++row) {
        addEntry(row);
      }
    }
  }
}

namespace {
std::optional<int32_t> fixedValueSize(const TypePtr& type) {
  if (type->isTimestamp()) {
    return sizeof(int64_t);
  }
  if (type->isFixedWidth()) {
    return type->cppSizeInBytes();
  }
  return std::nullopt;
}
} // namespace

// static
std::optional<int32_t> CompactRow::fixedRowSize(const RowTypePtr& rowType) {
  const size_t numFields = rowType->size();
  const size_t nullLength = bits::nbytes(numFields);

  size_t size = nullLength;
  for (const auto& child : rowType->children()) {
    if (auto valueBytes = fixedValueSize(child)) {
      size += valueBytes.value();
    } else {
      return std::nullopt;
    }
  }

  return size;
}

int32_t CompactRow::rowSize(vector_size_t index) const {
  return rowRowSize(index);
}

int32_t CompactRow::rowRowSize(vector_size_t index) const {
  const auto childIndex = decoded_.index(index);

  const auto numFields = children_.size();
  int32_t size = rowNullBytes_;

  for (auto i = 0; i < numFields; ++i) {
    if (childIsFixedWidth_[i]) {
      size += children_[i].valueBytes_;
    } else if (!children_[i].isNullAt(childIndex)) {
      size += children_[i].variableWidthRowSize(childIndex);
    }
  }

  return size;
}

void CompactRow::serializedRowSizes(
    const folly::Range<const vector_size_t*>& rows,
    vector_size_t** sizes) const {
  if (const auto fixedRowSize =
          row::CompactRow::fixedRowSize(asRowType(decoded_.base()->type()))) {
    for (const auto row : rows) {
      *sizes[row] = fixedRowSize.value() + sizeof(TRowSize);
    }
    return;
  }

  for (const auto& row : rows) {
    *sizes[row] = rowSize(row) + sizeof(TRowSize);
  }
}

int32_t CompactRow::serializeRow(vector_size_t index, char* buffer) const {
  auto childIndex = decoded_.index(index);

  int64_t valuesOffset = rowNullBytes_;

  auto* nulls = reinterpret_cast<uint8_t*>(buffer);

  for (auto i = 0; i < children_.size(); ++i) {
    auto& child = children_[i];

    // Write null bit. Advance offset if 'fixed-width'.
    if (child.isNullAt(childIndex)) {
      bits::setBit(nulls, i, true);
      if (childIsFixedWidth_[i]) {
        valuesOffset += child.valueBytes_;
      }
      continue;
    }

    if (childIsFixedWidth_[i]) {
      // Write fixed-width value.
      if (child.valueBytes_ > 0) {
        child.serializeFixedWidth(childIndex, buffer + valuesOffset);
      }
      valuesOffset += child.valueBytes_;
    } else {
      // Write non-null variable-width value.
      auto size =
          child.serializeVariableWidth(childIndex, buffer + valuesOffset);
      valuesOffset += size;
    }
  }

  return valuesOffset;
}

void CompactRow::serializeRow(
    vector_size_t offset,
    vector_size_t size,
    char* buffer,
    const size_t* bufferOffsets) const {
  raw_vector<vector_size_t> rows(size);
  raw_vector<uint8_t*> nulls(size);
  if (decoded_.isIdentityMapping()) {
    std::iota(rows.begin(), rows.end(), offset);
  } else {
    for (auto i = 0; i < size; ++i) {
      rows[i] = decoded_.index(offset + i);
    }
  }

  // After serializing each column, the 'offsets' are updated accordingly.
  std::vector<size_t> offsets(size);
  auto* const base = reinterpret_cast<uint8_t*>(buffer);
  for (auto i = 0; i < size; ++i) {
    nulls[i] = base + bufferOffsets[i];
    offsets[i] = bufferOffsets[i] + rowNullBytes_;
  }

  // Fixed-width and varchar/varbinary types are serialized using the vectorized
  // API 'serializedTyped'. Other data types are serialized row-by-row.
  for (auto childIdx = 0; childIdx < children_.size(); ++childIdx) {
    auto& child = children_[childIdx];
    if (childIsFixedWidth_[childIdx] ||
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
          buffer,
          offsets);
    } else {
      const bool mayHaveNulls = child.decoded_.mayHaveNulls();
      for (auto i = 0; i < rows.size(); ++i) {
        if (mayHaveNulls && child.isNullAt(rows[i])) {
          bits::setBit(nulls[i], childIdx, true);
        } else {
          // Write non-null variable-width value.
          offsets[i] +=
              child.serializeVariableWidth(rows[i], buffer + offsets[i]);
        }
      }
    }
  }
}

bool CompactRow::isNullAt(vector_size_t index) const {
  return decoded_.isNullAt(index);
}

int32_t CompactRow::variableWidthRowSize(vector_size_t index) const {
  switch (typeKind_) {
    case TypeKind::VARCHAR:
      [[fallthrough]];
    case TypeKind::VARBINARY: {
      const auto value = decoded_.valueAt<StringView>(index);
      return sizeof(int32_t) + value.size();
    }
    case TypeKind::ARRAY:
      return arrayRowSize(index);
    case TypeKind::MAP:
      return flatMap_ ? flatMapRowSize(index) : mapRowSize(index);
    case TypeKind::ROW:
      return rowRowSize(index);
    default:
      VELOX_UNREACHABLE(
          "Unexpected type kind: {}", TypeKindName::toName(typeKind_));
  };
}

int32_t CompactRow::arrayRowSize(vector_size_t index) const {
  const auto baseIndex = decoded_.index(index);

  auto* arrayBase = decoded_.base()->asUnchecked<ArrayVector>();
  const auto offset = arrayBase->offsetAt(baseIndex);
  const auto size = arrayBase->sizeAt(baseIndex);

  return arrayRowSize(children_[0], offset, size, childIsFixedWidth_[0]);
}

int32_t CompactRow::arrayRowSize(
    const CompactRow& elements,
    vector_size_t offset,
    vector_size_t size,
    bool fixedWidth) const {
  if (elements.typeKind_ == TypeKind::UNKNOWN) {
    return kSizeBytes;
  }

  const int32_t nullBytes = bits::nbytes(size);

  // array size | null bits | elements

  // 4 bytes for number of elements, some bytes for null flags.
  int32_t rowSize = sizeof(int32_t) + nullBytes;
  if (fixedWidth) {
    return rowSize + size * elements.valueBytes();
  }

  if (size == 0) {
    return rowSize;
  }

  // If element type is a complex type, then add 4 bytes for overall serialized
  // size of the array + 4 bytes per element for offset of the serialized
  // element.
  // size | nulls | serialized size | serialized offset 1 | serialized offset 2
  // |...| element 1 | element 2 |...

  if (!(elements.typeKind_ == TypeKind::VARCHAR ||
        elements.typeKind_ == TypeKind::VARBINARY)) {
    // 4 bytes for the overall serialized size + 4 bytes for the offset of each
    // element.
    rowSize += sizeof(int32_t) + size * sizeof(int32_t);
  }

  for (auto i = 0; i < size; ++i) {
    if (!elements.isNullAt(offset + i)) {
      rowSize += elements.variableWidthRowSize(offset + i);
    }
  }

  return rowSize;
}

int32_t CompactRow::serializeArray(vector_size_t index, char* buffer) const {
  auto baseIndex = decoded_.index(index);

  // For complex-type elements:
  // array size | null bits | serialized size | offset e1 | offset e2 |... | e1
  // | e2 |...
  //
  // 'serialized size' is the number of bytes starting after null bits and to
  // the end of the array. Offsets are specified relative to position right
  // after 'serialized size'.
  //
  // For fixed-width or string element type:
  // array size | null bite | e1 | e2 |...

  auto arrayBase = decoded_.base()->asUnchecked<ArrayVector>();
  auto offset = arrayBase->offsetAt(baseIndex);
  auto size = arrayBase->sizeAt(baseIndex);

  return serializeAsArray(
      children_[0], offset, size, childIsFixedWidth_[0], buffer);
}

int32_t CompactRow::serializeAsArray(
    const CompactRow& elements,
    vector_size_t offset,
    vector_size_t size,
    bool fixedWidth,
    char* buffer) const {
  // For complex-type elements:
  // array size | null bits | serialized size | offset e1 | offset e2 |... | e1
  // | e2 |...
  //
  // For fixed-width and string element types:
  // array size | null bits | e1 | e2 |...

  // Write array size.
  writeInt32(buffer, size);
  if (elements.typeKind_ == TypeKind::UNKNOWN) {
    return kSizeBytes;
  }

  // Write null flags.
  const int32_t nullBytes = bits::nbytes(size);
  const int32_t nullsOffset = kSizeBytes;

  int32_t elementsOffset = nullsOffset + nullBytes;

  auto* rawNulls = reinterpret_cast<uint8_t*>(buffer + nullsOffset);

  if (elements.supportsBulkCopy_) {
    if (elements.decoded_.mayHaveNulls()) {
      for (auto i = 0; i < size; ++i) {
        if (elements.isNullAt(offset + i)) {
          bits::setBit(rawNulls, i, true);
        }
      }
    }
    elements.serializeFixedWidth(offset, size, buffer + elementsOffset);
    return elementsOffset + size * elements.valueBytes_;
  }

  if (fixedWidth) {
    for (auto i = 0; i < size; ++i) {
      if (elements.isNullAt(offset + i)) {
        bits::setBit(rawNulls, i, true);
      } else {
        elements.serializeFixedWidth(offset + i, buffer + elementsOffset);
      }
      elementsOffset += elements.valueBytes_;
    }
  } else if (
      elements.typeKind_ == TypeKind::VARCHAR ||
      elements.typeKind_ == TypeKind::VARBINARY) {
    for (auto i = 0; i < size; ++i) {
      if (elements.isNullAt(offset + i)) {
        bits::setBit(rawNulls, i, true);
      } else {
        auto serializedBytes = elements.serializeVariableWidth(
            offset + i, buffer + elementsOffset);
        elementsOffset += serializedBytes;
      }
    }
  } else {
    if (size > 0) {
      // Leave room for serialized size and offsets.
      const size_t baseOffset = elementsOffset + kSizeBytes;
      elementsOffset += kSizeBytes + size * kSizeBytes;

      for (auto i = 0; i < size; ++i) {
        if (elements.isNullAt(offset + i)) {
          bits::setBit(rawNulls, i, true);
        } else {
          writeInt32(
              buffer + baseOffset + i * kSizeBytes,
              elementsOffset - baseOffset);

          auto serializedBytes = elements.serializeVariableWidth(
              offset + i, buffer + elementsOffset);

          elementsOffset += serializedBytes;
        }
      }

      writeInt32(buffer + baseOffset - kSizeBytes, elementsOffset - baseOffset);
    }
  }

  return elementsOffset;
}

int32_t CompactRow::mapRowSize(vector_size_t index) const {
  auto baseIndex = decoded_.index(index);

  //  <keys array> | <values array>

  auto* mapBase = decoded_.base()->asUnchecked<MapVector>();
  const auto offset = mapBase->offsetAt(baseIndex);
  const auto size = mapBase->sizeAt(baseIndex);

  return arrayRowSize(children_[0], offset, size, childIsFixedWidth_[0]) +
      arrayRowSize(children_[1], offset, size, childIsFixedWidth_[1]);
}

int32_t CompactRow::serializeMap(vector_size_t index, char* buffer) const {
  auto baseIndex = decoded_.index(index);

  //  <keys array> | <values array>

  auto mapBase = decoded_.base()->asUnchecked<MapVector>();
  auto offset = mapBase->offsetAt(baseIndex);
  auto size = mapBase->sizeAt(baseIndex);

  auto keysSerializedBytes = serializeAsArray(
      children_[0], offset, size, childIsFixedWidth_[0], buffer);

  auto valuesSerializedBytes = serializeAsArray(
      children_[1],
      offset,
      size,
      childIsFixedWidth_[1],
      buffer + keysSerializedBytes);

  return keysSerializedBytes + valuesSerializedBytes;
}

template <typename TElementAt>
int32_t CompactRow::entriesRowSize(
    vector_size_t size,
    bool fixedWidth,
    TElementAt elementAt) const {
  // Same layout as arrayRowSize().
  int32_t rowSize = sizeof(int32_t) + bits::nbytes(size);
  if (size == 0) {
    return rowSize;
  }
  const auto kind = elementAt(0).first->typeKind_;
  if (kind == TypeKind::UNKNOWN) {
    return kSizeBytes;
  }
  if (fixedWidth) {
    return rowSize + size * elementAt(0).first->valueBytes();
  }
  if (!(kind == TypeKind::VARCHAR || kind == TypeKind::VARBINARY)) {
    rowSize += sizeof(int32_t) + size * sizeof(int32_t);
  }
  for (auto i = 0; i < size; ++i) {
    const auto [element, index] = elementAt(i);
    if (!element->isNullAt(index)) {
      rowSize += element->variableWidthRowSize(index);
    }
  }
  return rowSize;
}

template <typename TElementAt>
int32_t CompactRow::serializeEntries(
    vector_size_t size,
    bool fixedWidth,
    TElementAt elementAt,
    char* buffer) const {
  // Same layout as serializeAsArray().
  writeInt32(buffer, size);
  const int32_t nullsOffset = kSizeBytes;
  int32_t elementsOffset = nullsOffset + bits::nbytes(size);
  if (size == 0) {
    return elementsOffset;
  }
  const auto kind = elementAt(0).first->typeKind_;
  if (kind == TypeKind::UNKNOWN) {
    return kSizeBytes;
  }
  auto* rawNulls = reinterpret_cast<uint8_t*>(buffer + nullsOffset);
  if (fixedWidth) {
    for (auto i = 0; i < size; ++i) {
      const auto [element, index] = elementAt(i);
      if (element->isNullAt(index)) {
        bits::setBit(rawNulls, i, true);
      } else {
        element->serializeFixedWidth(index, buffer + elementsOffset);
      }
      elementsOffset += element->valueBytes_;
    }
  } else if (kind == TypeKind::VARCHAR || kind == TypeKind::VARBINARY) {
    for (auto i = 0; i < size; ++i) {
      const auto [element, index] = elementAt(i);
      if (element->isNullAt(index)) {
        bits::setBit(rawNulls, i, true);
      } else {
        elementsOffset +=
            element->serializeVariableWidth(index, buffer + elementsOffset);
      }
    }
  } else {
    const size_t baseOffset = elementsOffset + kSizeBytes;
    elementsOffset += kSizeBytes + size * kSizeBytes;
    for (auto i = 0; i < size; ++i) {
      const auto [element, index] = elementAt(i);
      if (element->isNullAt(index)) {
        bits::setBit(rawNulls, i, true);
      } else {
        writeInt32(
            buffer + baseOffset + i * kSizeBytes, elementsOffset - baseOffset);
        elementsOffset +=
            element->serializeVariableWidth(index, buffer + elementsOffset);
      }
    }
    writeInt32(buffer + baseOffset - kSizeBytes, elementsOffset - baseOffset);
  }
  return elementsOffset;
}

int32_t CompactRow::flatMapRowSize(vector_size_t index) const {
  const auto baseIndex = decoded_.index(index);
  const auto begin = flatMapOffsets_[baseIndex];
  const auto size = flatMapOffsets_[baseIndex + 1] - begin;
  const auto keysRowSize =
      entriesRowSize(size, childIsFixedWidth_[0], [&](vector_size_t i) {
        return std::make_pair(&children_[0], flatMapChannels_[begin + i]);
      });
  if (!flatMapRawArrays_.empty()) {
    int32_t valuesRowSize = kSizeBytes + bits::nbytes(size);
    if (size == 0) {
      return keysRowSize + valuesRowSize;
    }
    valuesRowSize += kSizeBytes + size * kSizeBytes;
    const auto elementBytes =
        decoded_.base()->type()->childAt(1)->childAt(0)->cppSizeInBytes();
    for (auto i = 0; i < size; ++i) {
      const auto channel = flatMapChannels_[begin + i];
      if (const auto* array = flatMapRawArrays_[channel]) {
        if (!array->isNullAt(baseIndex)) {
          const auto numElements = array->sizeAt(baseIndex);
          valuesRowSize += kSizeBytes + bits::nbytes(numElements) +
              numElements * elementBytes;
        }
      } else {
        const auto& values = children_[flatMapValueChildren_[channel]];
        if (!values.isNullAt(baseIndex)) {
          valuesRowSize += values.variableWidthRowSize(baseIndex);
        }
      }
    }
    return keysRowSize + valuesRowSize;
  }
  if (!flatMapRawValues_.empty()) {
    return keysRowSize + kSizeBytes + bits::nbytes(size) +
        size * decoded_.base()->type()->childAt(1)->cppSizeInBytes();
  }
  return keysRowSize +
      entriesRowSize(size, childIsFixedWidth_[1], [&](vector_size_t i) {
           return std::make_pair(
               &children_[flatMapValueChildren_[flatMapChannels_[begin + i]]],
               baseIndex);
         });
}

namespace {

// Writes 'size' values of 'kBytes' bytes each in the layout of an array of
// fixed-width values. 'valueAt(i, out)' writes the i-th value to 'out' and
// returns false if the value is null. Returns number of bytes written.
template <int32_t kBytes, typename TValueAt>
int32_t
serializeFixedWidthEntries(vector_size_t size, TValueAt valueAt, char* buffer) {
  writeInt32(buffer, size);
  auto* rawNulls = reinterpret_cast<uint8_t*>(buffer + kSizeBytes);
  const int32_t valuesOffset = kSizeBytes + bits::nbytes(size);
  for (auto i = 0; i < size; ++i) {
    if (!valueAt(i, buffer + valuesOffset + i * kBytes)) {
      bits::setBit(rawNulls, i, true);
    }
  }
  return valuesOffset + size * kBytes;
}

// Calls 'func' with 'valueBytes' as a compile-time constant.
template <typename TFunc>
int32_t withValueBytes(int32_t valueBytes, TFunc func) {
  switch (valueBytes) {
    case 1:
      return func(std::integral_constant<int32_t, 1>{});
    case 2:
      return func(std::integral_constant<int32_t, 2>{});
    case 4:
      return func(std::integral_constant<int32_t, 4>{});
    case 8:
      return func(std::integral_constant<int32_t, 8>{});
    case 16:
      return func(std::integral_constant<int32_t, 16>{});
    default:
      VELOX_UNREACHABLE("Unexpected fixed-width value size: {}", valueBytes);
  }
}

} // namespace

int32_t CompactRow::serializeFlatMapRawArrays(
    vector_size_t baseIndex,
    vector_size_t begin,
    vector_size_t size,
    char* buffer) const {
  writeInt32(buffer, size);
  auto* rawNulls = reinterpret_cast<uint8_t*>(buffer + kSizeBytes);
  int32_t elementsOffset = kSizeBytes + bits::nbytes(size);
  if (size == 0) {
    return elementsOffset;
  }
  const size_t baseOffset = elementsOffset + kSizeBytes;
  elementsOffset += kSizeBytes + size * kSizeBytes;
  const auto elementBytes =
      decoded_.base()->type()->childAt(1)->childAt(0)->cppSizeInBytes();
  for (auto i = 0; i < size; ++i) {
    const auto channel = flatMapChannels_[begin + i];
    const auto* array = flatMapRawArrays_[channel];
    if (array == nullptr) {
      const auto& values = children_[flatMapValueChildren_[channel]];
      if (values.isNullAt(baseIndex)) {
        bits::setBit(rawNulls, i, true);
      } else {
        writeInt32(
            buffer + baseOffset + i * kSizeBytes, elementsOffset - baseOffset);
        elementsOffset +=
            values.serializeVariableWidth(baseIndex, buffer + elementsOffset);
      }
    } else if (array->isNullAt(baseIndex)) {
      bits::setBit(rawNulls, i, true);
    } else {
      writeInt32(
          buffer + baseOffset + i * kSizeBytes, elementsOffset - baseOffset);
      const auto offset = array->offsetAt(baseIndex);
      const auto numElements = array->sizeAt(baseIndex);
      writeInt32(buffer + elementsOffset, numElements);
      if (const auto* elementNulls = flatMapRawNulls_[channel]) {
        auto* rawElementNulls =
            reinterpret_cast<uint8_t*>(buffer + elementsOffset + kSizeBytes);
        for (auto j = 0; j < numElements; ++j) {
          if (bits::isBitNull(elementNulls, offset + j)) {
            bits::setBit(rawElementNulls, j, true);
          }
        }
      }
      elementsOffset += kSizeBytes + bits::nbytes(numElements);
      ::memcpy(
          buffer + elementsOffset,
          flatMapRawValues_[channel] + offset * elementBytes,
          numElements * elementBytes);
      elementsOffset += numElements * elementBytes;
    }
  }
  writeInt32(buffer + baseOffset - kSizeBytes, elementsOffset - baseOffset);
  return elementsOffset;
}

int32_t CompactRow::serializeFlatMap(vector_size_t index, char* buffer) const {
  const auto baseIndex = decoded_.index(index);
  const auto begin = flatMapOffsets_[baseIndex];
  const auto size = flatMapOffsets_[baseIndex + 1] - begin;
  const auto& keys = children_[0];
  const auto keysSerializedBytes =
      keys.supportsBulkCopy_ && keys.typeKind_ != TypeKind::UNKNOWN
      ? withValueBytes(
            keys.valueBytes_,
            [&](auto bytes) {
              constexpr int32_t kBytes = decltype(bytes)::value;
              const auto* rawKeys = keys.decoded_.data<char>();
              return serializeFixedWidthEntries<kBytes>(
                  size,
                  [&](vector_size_t i, char* out) {
                    ::memcpy(
                        out,
                        rawKeys + flatMapChannels_[begin + i] * kBytes,
                        kBytes);
                    return true;
                  },
                  buffer);
            })
      : serializeEntries(
            size,
            childIsFixedWidth_[0],
            [&](vector_size_t i) {
              return std::make_pair(
                  &keys,
                  static_cast<vector_size_t>(flatMapChannels_[begin + i]));
            },
            buffer);
  if (!flatMapRawArrays_.empty()) {
    return keysSerializedBytes +
        serializeFlatMapRawArrays(
               baseIndex, begin, size, buffer + keysSerializedBytes);
  }
  if (!flatMapRawValues_.empty()) {
    return keysSerializedBytes +
        withValueBytes(
               decoded_.base()->type()->childAt(1)->cppSizeInBytes(),
               [&](auto bytes) {
                 constexpr int32_t kBytes = decltype(bytes)::value;
                 return serializeFixedWidthEntries<kBytes>(
                     size,
                     [&](vector_size_t i, char* out) {
                       const auto channel = flatMapChannels_[begin + i];
                       if (const auto* rawValues = flatMapRawValues_[channel]) {
                         const auto* rawNulls = flatMapRawNulls_[channel];
                         if (rawNulls != nullptr &&
                             bits::isBitNull(rawNulls, baseIndex)) {
                           return false;
                         }
                         ::memcpy(out, rawValues + baseIndex * kBytes, kBytes);
                         return true;
                       }
                       const auto& values =
                           children_[flatMapValueChildren_[channel]];
                       if (values.isNullAt(baseIndex)) {
                         return false;
                       }
                       values.serializeFixedWidth(baseIndex, out);
                       return true;
                     },
                     buffer + keysSerializedBytes);
               });
  }
  return keysSerializedBytes +
      serializeEntries(
             size,
             childIsFixedWidth_[1],
             [&](vector_size_t i) {
               return std::make_pair(
                   &children_
                       [flatMapValueChildren_[flatMapChannels_[begin + i]]],
                   baseIndex);
             },
             buffer + keysSerializedBytes);
}

int32_t CompactRow::serialize(vector_size_t index, char* buffer) const {
  return serializeRow(index, buffer);
}

void CompactRow::serialize(
    vector_size_t offset,
    vector_size_t size,
    const size_t* bufferOffsets,
    char* buffer) const {
  serializeRow(offset, size, buffer, bufferOffsets);
}

void CompactRow::serializeFixedWidth(vector_size_t index, char* buffer) const {
  VELOX_DCHECK(fixedWidthTypeKind_);
  switch (typeKind_) {
    case TypeKind::BOOLEAN:
      *reinterpret_cast<bool*>(buffer) = decoded_.valueAt<bool>(index);
      break;
    case TypeKind::TIMESTAMP: {
      auto micros = decoded_.valueAt<Timestamp>(index).toMicros();
      ::memcpy(buffer, &micros, sizeof(int64_t));
      break;
    }
    default:
      ::memcpy(
          buffer,
          decoded_.data<char>() + decoded_.index(index) * valueBytes_,
          valueBytes_);
  }
}

void CompactRow::serializeFixedWidth(
    vector_size_t offset,
    vector_size_t size,
    char* buffer) const {
  VELOX_DCHECK(supportsBulkCopy_);
  // decoded_.data<char>() can be null if all values are null.
  if (decoded_.data<char>()) {
    ::memcpy(
        buffer,
        decoded_.data<char>() + decoded_.index(offset) * valueBytes_,
        valueBytes_ * size);
  }
}

int32_t CompactRow::serializeVariableWidth(vector_size_t index, char* buffer)
    const {
  switch (typeKind_) {
    case TypeKind::VARCHAR:
      [[fallthrough]];
    case TypeKind::VARBINARY: {
      auto value = decoded_.valueAt<StringView>(index);
      writeInt32(buffer, value.size());
      if (!value.empty()) {
        ::memcpy(buffer + kSizeBytes, value.data(), value.size());
      }
      return kSizeBytes + value.size();
    }
    case TypeKind::ARRAY:
      return serializeArray(index, buffer);
    case TypeKind::MAP:
      return flatMap_ ? serializeFlatMap(index, buffer)
                      : serializeMap(index, buffer);
    case TypeKind::ROW:
      return serializeRow(index, buffer);
    default:
      VELOX_UNREACHABLE(
          "Unexpected type kind: {}", TypeKindName::toName(typeKind_));
  };
}

namespace {

// Reads single fixed-width value from buffer into flatVector[index].
template <typename T>
void readFixedWidthValue(
    bool isNull,
    const char* buffer,
    FlatVector<T>* flatVector,
    vector_size_t index) {
  if (isNull) {
    flatVector->setNull(index, true);
  } else if constexpr (std::is_same_v<T, Timestamp>) {
    int64_t micros;
    ::memcpy(&micros, buffer, sizeof(int64_t));
    flatVector->set(index, Timestamp::fromMicros(micros));
  } else {
    T value;
    ::memcpy(&value, buffer, sizeof(T));
    flatVector->set(index, value);
  }
}

template <typename T>
int32_t valueSize() {
  if constexpr (std::is_same_v<T, Timestamp>) {
    return sizeof(int64_t);
  } else {
    return sizeof(T);
  }
}

// Deserializes one fixed-width value from each 'row' in 'data'.
// Each value starts at data[row].data() + offsets[row].
//
// @param nulls Null flags for the values.
// @param offsets Offsets in 'data' for the serialized values. Not used if value
// is null.
template <TypeKind Kind>
VectorPtr deserializeFixedWidth(
    const TypePtr& type,
    const std::vector<std::string_view>& data,
    const BufferPtr& nulls,
    const std::vector<size_t>& offsets,
    memory::MemoryPool* pool) {
  using T = typename TypeTraits<Kind>::NativeType;

  const auto numRows = data.size();
  auto flatVector = BaseVector::create<FlatVector<T>>(type, numRows, pool);

  auto* rawNulls = nulls->as<uint64_t>();

  for (auto i = 0; i < numRows; ++i) {
    const bool isNull = bits::isBitNull(rawNulls, i);
    readFixedWidthValue<T>(
        isNull,
        isNull ? nullptr : data[i].data() + offsets[i],
        flatVector.get(),
        i);
  }

  return flatVector;
}

vector_size_t totalSize(const vector_size_t* rawSizes, size_t numRows) {
  vector_size_t total = 0;
  for (auto i = 0; i < numRows; ++i) {
    total += rawSizes[i];
  }
  return total;
}

const uint8_t* readNulls(const char* buffer) {
  return reinterpret_cast<const uint8_t*>(buffer);
}

// Deserializes multiple fixed-width values from each 'row' in 'data'.
// Each set of values starts at data[row].data() + offsets[row] and contains
// null flags followed by values. The number of values is provided in
// sizes[row].
// nulls | v1 | v2 | v3 |...
// Advances offsets past the last value.
template <TypeKind Kind>
VectorPtr deserializeFixedWidthArrays(
    const TypePtr& type,
    const std::vector<std::string_view>& data,
    const BufferPtr& sizes,
    std::vector<size_t>& offsets,
    memory::MemoryPool* pool) {
  using T = typename TypeTraits<Kind>::NativeType;

  const int32_t valueBytes = valueSize<T>();

  const auto numRows = data.size();
  auto* rawSizes = sizes->as<vector_size_t>();

  const auto total = totalSize(rawSizes, numRows);

  auto flatVector = BaseVector::create<FlatVector<T>>(type, total, pool);

  vector_size_t index = 0;
  for (auto i = 0; i < numRows; ++i) {
    const auto size = rawSizes[i];
    if (size > 0) {
      auto nullBytes = bits::nbytes(size);

      auto* rawElementNulls = readNulls(data[i].data() + offsets[i]);

      offsets[i] += nullBytes;

      for (auto j = 0; j < size; ++j) {
        readFixedWidthValue<T>(
            bits::isBitSet(rawElementNulls, j),
            data[i].data() + offsets[i],
            flatVector.get(),
            index);

        offsets[i] += valueBytes;
        ++index;
      }
    }
  }

  return flatVector;
}

int32_t readString(
    const char* buffer,
    FlatVector<StringView>* flatVector,
    vector_size_t index) {
  int32_t size = readInt32(buffer);
  StringView value(buffer + kSizeBytes, size);
  flatVector->set(index, value);
  return kSizeBytes + size;
}

VectorPtr deserializeUnknowns(
    const TypePtr& type,
    const std::vector<std::string_view>& data,
    const BufferPtr& nulls,
    std::vector<size_t>& offsets,
    memory::MemoryPool* pool) {
  return BaseVector::createNullConstant(UNKNOWN(), data.size(), pool);
}

// Deserializes one string from each 'row' in 'data'.
// Each strings starts at data[row].data() + offsets[row].
// string size | <string bytes>
// Advances the offsets past the strings.
VectorPtr deserializeStrings(
    const TypePtr& type,
    const std::vector<std::string_view>& data,
    const BufferPtr& nulls,
    std::vector<size_t>& offsets,
    memory::MemoryPool* pool) {
  const auto numRows = data.size();
  auto flatVector =
      BaseVector::create<FlatVector<StringView>>(type, numRows, pool);

  auto* rawNulls = nulls->as<uint64_t>();

  for (auto i = 0; i < numRows; ++i) {
    if (bits::isBitNull(rawNulls, i)) {
      flatVector->setNull(i, true);
    } else {
      offsets[i] +=
          readString(data[i].data() + offsets[i], flatVector.get(), i);
    }
  }

  return flatVector;
}

// Deserializes multiple strings from each 'row' in 'data'.
// Each set of strings starts at data[row].data() + offsets[row] and contains
// null flags followed by the strings. The number of strings is provided in
// sizes[row].
// nulls | size-of-s1 | <s1> | size-of-s2 | <s2> |...
// Advances offsets past the last string.
VectorPtr deserializeStringArrays(
    const TypePtr& type,
    const std::vector<std::string_view>& data,
    const BufferPtr& sizes,
    std::vector<size_t>& offsets,
    memory::MemoryPool* pool) {
  const auto numRows = data.size();
  auto* rawSizes = sizes->as<vector_size_t>();

  const auto total = totalSize(rawSizes, numRows);

  auto flatVector =
      BaseVector::create<FlatVector<StringView>>(type, total, pool);

  vector_size_t index = 0;
  for (auto i = 0; i < numRows; ++i) {
    const auto size = rawSizes[i];
    if (size > 0) {
      auto nullBytes = bits::nbytes(size);

      auto* rawElementNulls = readNulls(data[i].data() + offsets[i]);

      offsets[i] += nullBytes;

      for (auto j = 0; j < size; ++j) {
        if (bits::isBitSet(rawElementNulls, j)) {
          flatVector->setNull(index++, true);
        } else {
          offsets[i] +=
              readString(data[i].data() + offsets[i], flatVector.get(), index);
          ++index;
        }
      }
    }
  }

  return flatVector;
}

VectorPtr deserialize(
    const TypePtr& type,
    const std::vector<std::string_view>& data,
    const BufferPtr& nulls,
    std::vector<size_t>& offsets,
    memory::MemoryPool* pool);

// Deserializes multiple arrays from each 'row' in 'data'.
// Each set of arrays starts at data[row].data() + offsets[row] and contains
// null flags followed by the arrays. The number of arrays is provided in
// sizes[row].
// nulls | serializes size | offset-of-a1 | offset-of-a2 |...
// |size-of-a1 | nulls-of-a1-elements | <a1 elements> |...
//
// Advances offsets past the last array.
VectorPtr deserializeComplexArrays(
    const TypePtr& type,
    const std::vector<std::string_view>& data,
    const BufferPtr& sizes,
    std::vector<size_t>& offsets,
    memory::MemoryPool* pool) {
  const auto numRows = data.size();
  auto* rawSizes = sizes->as<vector_size_t>();

  const auto total = totalSize(rawSizes, numRows);

  BufferPtr nulls = allocateNulls(total, pool);
  auto* rawNulls = nulls->asMutable<uint64_t>();

  std::vector<std::string_view> nestedData(total);
  std::vector<size_t> nestedOffsets(total, 0);

  vector_size_t nestedIndex = 0;
  for (auto i = 0; i < numRows; ++i) {
    const auto size = rawSizes[i];
    if (size > 0) {
      // Read nulls.
      auto* rawElementNulls = readNulls(data[i].data() + offsets[i]);
      offsets[i] += bits::nbytes(size);

      // Read serialized size.
      auto serializedSize = readInt32(data[i].data() + offsets[i]);
      offsets[i] += kSizeBytes;

      // Read offsets of individual elements.
      auto buffer = data[i].data() + offsets[i];
      for (auto j = 0; j < size; ++j) {
        if (bits::isBitSet(rawElementNulls, j)) {
          bits::setNull(rawNulls, nestedIndex);
        } else {
          int32_t nestedOffset = readInt32(buffer + j * kSizeBytes);
          nestedOffsets[nestedIndex] = offsets[i] + nestedOffset;
          nestedData[nestedIndex] = data[i];
        }
        ++nestedIndex;
      }

      offsets[i] += serializedSize;
    }
  }

  return deserialize(type, nestedData, nulls, nestedOffsets, pool);
}

// Deserializes one array from each 'row' in 'data'.
// Each array starts at data[row].data() + offsets[row].
// size | element nulls | serialized size (if complex type elements)
// | element offsets (if complex type elements) | e1 | e2 | e3 |...
//
// UNKNOWN elements have no null flags or value bytes.
// Advances the offsets past the arrays.
ArrayVectorPtr deserializeArrays(
    const TypePtr& type,
    const std::vector<std::string_view>& data,
    const BufferPtr& nulls,
    std::vector<size_t>& offsets,
    memory::MemoryPool* pool) {
  const auto numRows = data.size();

  auto* rawNulls = nulls->as<uint64_t>();

  BufferPtr arrayOffsets = allocateOffsets(numRows, pool);
  auto* rawArrayOffsets = arrayOffsets->asMutable<vector_size_t>();

  BufferPtr arraySizes = allocateSizes(numRows, pool);
  auto* rawArraySizes = arraySizes->asMutable<vector_size_t>();

  vector_size_t arrayOffset = 0;

  for (auto i = 0; i < numRows; ++i) {
    if (!bits::isBitNull(rawNulls, i)) {
      // Read array size.
      int32_t size = readInt32(data[i].data() + offsets[i]);
      offsets[i] += kSizeBytes;

      rawArrayOffsets[i] = arrayOffset;
      rawArraySizes[i] = size;
      arrayOffset += size;
    }
  }

  VectorPtr elements;
  const auto& elementType = type->childAt(0);
  if (elementType->isUnknown()) {
    elements = BaseVector::createNullConstant(elementType, arrayOffset, pool);
  } else if (elementType->isFixedWidth()) {
    elements = VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
        deserializeFixedWidthArrays,
        elementType->kind(),
        elementType,
        data,
        arraySizes,
        offsets,
        pool);
  } else {
    switch (elementType->kind()) {
      case TypeKind::VARCHAR:
      case TypeKind::VARBINARY:
        elements = deserializeStringArrays(
            elementType, data, arraySizes, offsets, pool);
        break;
      case TypeKind::ARRAY:
      case TypeKind::MAP:
      case TypeKind::ROW:
        elements = deserializeComplexArrays(
            elementType, data, arraySizes, offsets, pool);
        break;
      default:
        VELOX_UNREACHABLE("{}", elementType->toString());
    }
  }

  return std::make_shared<ArrayVector>(
      pool, type, nulls, numRows, arrayOffsets, arraySizes, elements);
}

// Deserializes one map from each 'row' in 'data'.
// Each map starts at data[row].data() + offsets[row].
// array-of-keys | array-of-values
// Advances the offsets past the maps.
VectorPtr deserializeMaps(
    const TypePtr& type,
    const std::vector<std::string_view>& data,
    const BufferPtr& nulls,
    std::vector<size_t>& offsets,
    memory::MemoryPool* pool) {
  auto arrayOfKeysType = ARRAY(type->childAt(0));
  auto arrayOfValuesType = ARRAY(type->childAt(1));
  auto arrayOfKeys =
      deserializeArrays(arrayOfKeysType, data, nulls, offsets, pool);
  auto arrayOfValues =
      deserializeArrays(arrayOfValuesType, data, nulls, offsets, pool);

  return std::make_shared<MapVector>(
      pool,
      type,
      nulls,
      data.size(),
      arrayOfKeys->offsets(),
      arrayOfKeys->sizes(),
      arrayOfKeys->elements(),
      arrayOfValues->elements());
}

RowVectorPtr deserializeRows(
    const TypePtr& type,
    const std::vector<std::string_view>& data,
    const BufferPtr& nulls,
    std::vector<size_t>& offsets,
    memory::MemoryPool* pool);

// Switches on 'type' and calls type-specific deserialize method to deserialize
// one value from each 'row' in 'data' starting at the specified offset.
// Each value starts at data[row].data() + offsets[row].
// If 'type' is variable-width, advances 'offsets' past deserialized values.
// If 'type' is fixed-width, offsets remain unmodified.
VectorPtr deserialize(
    const TypePtr& type,
    const std::vector<std::string_view>& data,
    const BufferPtr& nulls,
    std::vector<size_t>& offsets,
    memory::MemoryPool* pool) {
  const auto typeKind = type->kind();

  if (typeKind == TypeKind::UNKNOWN) {
    return deserializeUnknowns(type, data, nulls, offsets, pool);
  }

  if (type->isFixedWidth()) {
    return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
        deserializeFixedWidth, typeKind, type, data, nulls, offsets, pool);
  }
  switch (typeKind) {
    case TypeKind::VARCHAR:
    case TypeKind::VARBINARY:
      return deserializeStrings(type, data, nulls, offsets, pool);
    case TypeKind::ARRAY:
      return deserializeArrays(type, data, nulls, offsets, pool);
    case TypeKind::MAP:
      return deserializeMaps(type, data, nulls, offsets, pool);
    case TypeKind::ROW:
      return deserializeRows(type, data, nulls, offsets, pool);
    default:
      VELOX_UNREACHABLE("{}", type->toString());
  }
}

// Deserializes one struct from each 'row' in 'data'.
// nulls | field1 | field2 |...
RowVectorPtr deserializeRows(
    const TypePtr& type,
    const std::vector<std::string_view>& data,
    const BufferPtr& nulls,
    std::vector<size_t>& offsets,
    memory::MemoryPool* pool) {
  const auto numRows = data.size();
  const size_t numFields = type->size();

  std::vector<VectorPtr> fields;

  auto* rawNulls = nulls != nullptr ? nulls->as<uint64_t>() : nullptr;

  std::vector<BufferPtr> fieldNulls;
  fieldNulls.reserve(numFields);
  for (auto i = 0; i < numFields; ++i) {
    fieldNulls.emplace_back(allocateNulls(numRows, pool));
    auto* rawFieldNulls = fieldNulls.back()->asMutable<uint8_t>();
    for (auto row = 0; row < numRows; ++row) {
      // Callers provide one offset per input row.
      const auto isNull =
          (rawNulls != nullptr && bits::isBitNull(rawNulls, row)) ||
          // NOLINTNEXTLINE(facebook-hte-ParameterUncheckedArrayBounds)
          bits::isBitSet(readNulls(data[row].data() + offsets[row]), i);
      bits::setBit(rawFieldNulls, row, !isNull);
    }
  }

  const size_t nullLength = bits::nbytes(numFields);
  for (auto row = 0; row < numRows; ++row) {
    if (rawNulls != nullptr && bits::isBitNull(rawNulls, row)) {
      continue;
    }
    offsets[row] += nullLength;
  }

  for (auto i = 0; i < numFields; ++i) {
    const auto& child = type->childAt(i);
    auto field = deserialize(child, data, fieldNulls[i], offsets, pool);
    // If 'field' is fixed-width, advance offsets for rows where top-level
    // struct is not null.
    if (auto numBytes = fixedValueSize(child)) {
      for (auto row = 0; row < numRows; ++row) {
        const auto isTopLevelNull =
            rawNulls != nullptr && bits::isBitNull(rawNulls, row);
        if (!isTopLevelNull) {
          offsets[row] += numBytes.value();
        }
      }
    }
    fields.emplace_back(std::move(field));
  }

  return std::make_shared<RowVector>(
      pool, type, nulls, numRows, std::move(fields));
}

} // namespace

// static
RowVectorPtr CompactRow::deserialize(
    const std::vector<std::string_view>& data,
    const RowTypePtr& rowType,
    memory::MemoryPool* pool) {
  return deserialize(data, nullptr, rowType, pool);
}

// static
RowVectorPtr CompactRow::deserialize(
    const std::vector<std::string_view>& data,
    const BufferPtr& nulls,
    const RowTypePtr& rowType,
    memory::MemoryPool* pool) {
  const auto numRows = data.size();
  std::vector<size_t> offsets(numRows, 0);

  return deserializeRows(rowType, data, nulls, offsets, pool);
}

} // namespace facebook::velox::row
