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
#include "velox/vector/FlatVector.h"

namespace facebook::velox::row {

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
      auto arrayBase = base->as<ArrayVector>();
      children_.push_back(CompactRow(arrayBase->elements()));
      childIsFixedWidth_.push_back(
          arrayBase->elements()->type()->isFixedWidth());
      break;
    }
    case TypeKind::MAP: {
      auto mapBase = base->as<MapVector>();
      children_.push_back(CompactRow(mapBase->mapKeys()));
      children_.push_back(CompactRow(mapBase->mapValues()));
      childIsFixedWidth_.push_back(mapBase->mapKeys()->type()->isFixedWidth());
      childIsFixedWidth_.push_back(
          mapBase->mapValues()->type()->isFixedWidth());
      break;
    }
    case TypeKind::ROW: {
      auto rowBase = base->as<RowVector>();
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
      FOLLY_FALLTHROUGH;
    case TypeKind::SMALLINT:
      FOLLY_FALLTHROUGH;
    case TypeKind::INTEGER:
      FOLLY_FALLTHROUGH;
    case TypeKind::BIGINT:
      FOLLY_FALLTHROUGH;
    case TypeKind::REAL:
      FOLLY_FALLTHROUGH;
    case TypeKind::DOUBLE:
      FOLLY_FALLTHROUGH;
    case TypeKind::UNKNOWN:
      valueBytes_ = type->cppSizeInBytes();
      fixedWidthTypeKind_ = true;
      supportsBulkCopy_ = decoded_.isIdentityMapping();
      break;
    case TypeKind::TIMESTAMP:
      valueBytes_ = sizeof(int64_t);
      fixedWidthTypeKind_ = true;
      break;
    case TypeKind::VARCHAR:
      FOLLY_FALLTHROUGH;
    case TypeKind::VARBINARY:
      // Nothing to do.
      break;
    default:
      VELOX_UNSUPPORTED("Unsupported type: {}", type->toString());
  }
}

// static
std::optional<int32_t> CompactRow::fixedRowSize(const RowTypePtr& rowType) {
  const size_t numFields = rowType->size();
  const size_t nullLength = bits::nbytes(numFields);

  size_t size = nullLength;
  for (const auto& child : rowType->children()) {
    if (child->isTimestamp()) {
      size += sizeof(int64_t);
    } else if (child->isFixedWidth()) {
      size += child->cppSizeInBytes();
    } else {
      return std::nullopt;
    }
  }

  return size;
}

int32_t CompactRow::rowSize(vector_size_t index) {
  return rowRowSize(index);
}

int32_t CompactRow::rowRowSize(vector_size_t index) {
  auto childIndex = decoded_.index(index);

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

int32_t CompactRow::serializeRow(vector_size_t index, char* buffer) {
  auto childIndex = decoded_.index(index);

  int64_t valuesOffset = rowNullBytes_;

  auto* nulls = reinterpret_cast<uint8_t*>(buffer);

  for (auto i = 0; i < children_.size(); ++i) {
    auto& child = children_[i];

    // Write fixed-width value.
    if (childIsFixedWidth_[i]) {
      child.serializeFixedWidth(childIndex, buffer + valuesOffset);
      valuesOffset += child.valueBytes_;
    }

    // Write null bit.
    if (child.isNullAt(childIndex)) {
      bits::setBit(nulls, i, true);
      continue;
    }

    // Write non-null variable-width value.
    if (!childIsFixedWidth_[i]) {
      auto size =
          child.serializeVariableWidth(childIndex, buffer + valuesOffset);
      valuesOffset += size;
    }
  }

  return valuesOffset;
}

bool CompactRow::isNullAt(vector_size_t index) {
  return decoded_.isNullAt(index);
}

int32_t CompactRow::variableWidthRowSize(vector_size_t index) {
  switch (typeKind_) {
    case TypeKind::VARCHAR:
      FOLLY_FALLTHROUGH;
    case TypeKind::VARBINARY: {
      auto value = decoded_.valueAt<StringView>(index);
      return sizeof(int32_t) + value.size();
    }
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

int32_t CompactRow::arrayRowSize(vector_size_t index) {
  auto baseIndex = decoded_.index(index);

  // array size | null bits | elements
  auto arrayBase = decoded_.base()->asUnchecked<ArrayVector>();
  auto offset = arrayBase->offsetAt(baseIndex);
  auto size = arrayBase->sizeAt(baseIndex);

  return arrayRowSize(children_[0], offset, size, childIsFixedWidth_[0]);
}

int32_t CompactRow::arrayRowSize(
    CompactRow& elements,
    vector_size_t offset,
    vector_size_t size,
    bool fixedWidth) {
  const int32_t nullBytes = bits::nbytes(size);

  int32_t rowSize = sizeof(int32_t) + nullBytes;
  if (fixedWidth) {
    return rowSize + size * elements.valueBytes();
  }

  for (auto i = 0; i < size; ++i) {
    if (!elements.isNullAt(offset + i)) {
      rowSize += elements.variableWidthRowSize(offset + i);
    }
  }

  return rowSize;
}

int32_t CompactRow::serializeArray(vector_size_t index, char* buffer) {
  auto baseIndex = decoded_.index(index);

  // array size | null bits | elements
  auto arrayBase = decoded_.base()->asUnchecked<ArrayVector>();
  auto offset = arrayBase->offsetAt(baseIndex);
  auto size = arrayBase->sizeAt(baseIndex);

  return serializeAsArray(
      children_[0], offset, size, childIsFixedWidth_[0], buffer);
}

int32_t CompactRow::serializeAsArray(
    CompactRow& elements,
    vector_size_t offset,
    vector_size_t size,
    bool fixedWidth,
    char* buffer) {
  // array size | null bits | elements

  // Write array size.
  *reinterpret_cast<int32_t*>(buffer) = size;

  const int32_t nullBytes = bits::nbytes(size);
  const int32_t nullsOffset = sizeof(int32_t);

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

  for (auto i = 0; i < size; ++i) {
    if (elements.isNullAt(offset + i)) {
      bits::setBit(rawNulls, i, true);
      if (fixedWidth) {
        elementsOffset += elements.valueBytes_;
      }
    } else {
      if (fixedWidth) {
        elements.serializeFixedWidth(offset + i, buffer + elementsOffset);
        elementsOffset += elements.valueBytes_;
      } else {
        auto serializedBytes = elements.serializeVariableWidth(
            offset + i, buffer + elementsOffset);

        elementsOffset += serializedBytes;
      }
    }
  }
  return elementsOffset;
}

int32_t CompactRow::mapRowSize(vector_size_t index) {
  auto baseIndex = decoded_.index(index);

  //  <keys array> | <values array>

  auto mapBase = decoded_.base()->asUnchecked<MapVector>();
  auto offset = mapBase->offsetAt(baseIndex);
  auto size = mapBase->sizeAt(baseIndex);

  return arrayRowSize(children_[0], offset, size, childIsFixedWidth_[0]) +
      arrayRowSize(children_[1], offset, size, childIsFixedWidth_[1]);
}

int32_t CompactRow::serializeMap(vector_size_t index, char* buffer) {
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

int32_t CompactRow::serialize(vector_size_t index, char* buffer) {
  return serializeRow(index, buffer);
}

void CompactRow::serializeFixedWidth(vector_size_t index, char* buffer) {
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

void CompactRow::serializeFixedWidth(
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

int32_t CompactRow::serializeVariableWidth(vector_size_t index, char* buffer) {
  switch (typeKind_) {
    case TypeKind::VARCHAR:
      FOLLY_FALLTHROUGH;
    case TypeKind::VARBINARY: {
      auto value = decoded_.valueAt<StringView>(index);
      *reinterpret_cast<int32_t*>(buffer) = value.size();
      if (!value.empty()) {
        memcpy(buffer + sizeof(int32_t), value.data(), value.size());
      }
      return sizeof(int32_t) + value.size();
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

namespace {

/// @param nulls Null flags for the values.
/// @param offsets In/out parameter that specifies offsets in 'data' for the
/// serialized values. Advanced past the serialized value.
template <TypeKind Kind>
VectorPtr deserializeFixedWidth(
    const TypePtr& type,
    const std::vector<std::string_view>& data,
    const BufferPtr& nulls,
    std::vector<size_t>& offsets,
    memory::MemoryPool* pool) {
  using T = typename TypeTraits<Kind>::NativeType;

  const auto numRows = data.size();
  auto flatVector = BaseVector::create<FlatVector<T>>(type, numRows, pool);

  auto* rawNulls = nulls->as<uint64_t>();

  for (auto i = 0; i < numRows; ++i) {
    if (bits::isBitNull(rawNulls, i)) {
      flatVector->setNull(i, true);
    } else if constexpr (std::is_same_v<T, Timestamp>) {
      int64_t micros =
          *reinterpret_cast<const int64_t*>(data[i].data() + offsets[i]);
      flatVector->set(i, Timestamp::fromMicros(micros));
    } else {
      T value = *reinterpret_cast<const T*>(data[i].data() + offsets[i]);
      flatVector->set(i, value);
    }

    if constexpr (std::is_same_v<T, Timestamp>) {
      offsets[i] += sizeof(int64_t);
    } else {
      offsets[i] += sizeof(T);
    }
  }

  return flatVector;
}

template <TypeKind Kind>
VectorPtr deserializeFixedWidthArrays(
    const TypePtr& type,
    const std::vector<std::string_view>& data,
    const BufferPtr& sizes,
    std::vector<size_t>& offsets,
    memory::MemoryPool* pool) {
  using T = typename TypeTraits<Kind>::NativeType;

  const auto numRows = data.size();
  auto* rawSizes = sizes->as<vector_size_t>();

  vector_size_t total = 0;
  for (auto i = 0; i < numRows; ++i) {
    total += rawSizes[i];
  }

  auto flatVector = BaseVector::create<FlatVector<T>>(type, total, pool);

  vector_size_t index = 0;
  for (auto i = 0; i < numRows; ++i) {
    const auto size = rawSizes[i];
    if (size > 0) {
      auto nullBytes = bits::nbytes(size);

      auto* rawElementNulls =
          reinterpret_cast<const uint8_t*>(data[i].data() + offsets[i]);

      offsets[i] += nullBytes;

      for (auto j = 0; j < size; ++j) {
        if (bits::isBitSet(rawElementNulls, j)) {
          flatVector->setNull(index++, true);
        } else {
          T value = *reinterpret_cast<const T*>(data[i].data() + offsets[i]);
          flatVector->set(index++, value);
        }
        offsets[i] += sizeof(T);
      }
    }
  }

  return flatVector;
}

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
      auto* buffer = data[i].data() + offsets[i];
      int32_t size = *reinterpret_cast<const int32_t*>(buffer);
      StringView value(buffer + sizeof(int32_t), size);
      flatVector->set(i, value);
      offsets[i] += sizeof(int32_t) + size;
    }
  }

  return flatVector;
}

VectorPtr deserializeStringArrays(
    const TypePtr& type,
    const std::vector<std::string_view>& data,
    const BufferPtr& sizes,
    std::vector<size_t>& offsets,
    memory::MemoryPool* pool) {
  const auto numRows = data.size();
  auto* rawSizes = sizes->as<vector_size_t>();

  vector_size_t total = 0;
  for (auto i = 0; i < numRows; ++i) {
    total += rawSizes[i];
  }

  auto flatVector =
      BaseVector::create<FlatVector<StringView>>(type, total, pool);

  vector_size_t index = 0;
  for (auto i = 0; i < numRows; ++i) {
    const auto size = rawSizes[i];
    if (size > 0) {
      auto nullBytes = bits::nbytes(size);

      auto* rawElementNulls =
          reinterpret_cast<const uint8_t*>(data[i].data() + offsets[i]);

      offsets[i] += nullBytes;

      for (auto j = 0; j < size; ++j) {
        if (bits::isBitSet(rawElementNulls, j)) {
          flatVector->setNull(index++, true);
        } else {
          auto* buffer = data[i].data() + offsets[i];
          int32_t stringSize = *reinterpret_cast<const int32_t*>(buffer);
          StringView value(buffer + sizeof(int32_t), stringSize);
          flatVector->set(index++, value);
          offsets[i] += sizeof(int32_t) + stringSize;
        }
      }
    }
  }

  return flatVector;
}

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
      auto* buffer = data[i].data() + offsets[i];
      int32_t size = *reinterpret_cast<const int32_t*>(buffer);

      offsets[i] += sizeof(int32_t);

      rawArrayOffsets[i] = arrayOffset;
      rawArraySizes[i] = size;
      arrayOffset += size;
    }
  }

  VectorPtr elements;
  const auto& elementType = type->childAt(0);
  if (elementType->isFixedWidth()) {
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
      default:
        VELOX_NYI();
    }
  }

  return std::make_shared<ArrayVector>(
      pool, type, nulls, numRows, arrayOffsets, arraySizes, elements);
}

VectorPtr deserializeMaps(
    const TypePtr& type,
    const std::vector<std::string_view>& data,
    const BufferPtr& nulls,
    std::vector<size_t>& offsets,
    memory::MemoryPool* pool) {
  auto arrayOfKeys =
      deserializeArrays(ARRAY(type->childAt(0)), data, nulls, offsets, pool);
  auto arrayOfValues =
      deserializeArrays(ARRAY(type->childAt(1)), data, nulls, offsets, pool);

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
    memory::MemoryPool* pool) {
  const auto numRows = data.size();
  const size_t numFields = type->size();
  const size_t nullLength = bits::nbytes(numFields);

  std::vector<VectorPtr> fields;

  auto* rawRowNulls = nulls != nullptr ? nulls->as<uint64_t>() : nullptr;

  std::vector<BufferPtr> fieldNulls;
  fieldNulls.reserve(numFields);
  for (auto i = 0; i < numFields; ++i) {
    fieldNulls.emplace_back(allocateNulls(numRows, pool));
    auto* rawFieldNulls = fieldNulls.back()->asMutable<uint8_t>();
    for (auto row = 0; row < numRows; ++row) {
      auto* serializedNulls = data[row].data() + offsets[row];
      bits::setBit(
          rawFieldNulls,
          row,
          (rawRowNulls != nullptr && bits::isBitNull(rawRowNulls, row)) ||
              !bits::isBitSet(serializedNulls, i));
    }
  }

  for (auto row = 0; row < numRows; ++row) {
    offsets[row] += nullLength;
  }

  for (auto i = 0; i < numFields; ++i) {
    VectorPtr field;

    const auto typeKind = type->childAt(i)->kind();

    if (type->childAt(i)->isFixedWidth()) {
      field = VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
          deserializeFixedWidth,
          type->childAt(i)->kind(),
          type->childAt(i),
          data,
          fieldNulls[i],
          offsets,
          pool);
    } else {
      switch (typeKind) {
        case TypeKind::VARCHAR:
        case TypeKind::VARBINARY:
          field = deserializeStrings(
              type->childAt(i), data, fieldNulls[i], offsets, pool);
          break;
        case TypeKind::ARRAY:
          field = deserializeArrays(
              type->childAt(i), data, fieldNulls[i], offsets, pool);
          break;
        case TypeKind::MAP:
          field = deserializeMaps(
              type->childAt(i), data, fieldNulls[i], offsets, pool);
          break;
        case TypeKind::ROW:
          field = deserializeRows(
              type->childAt(i), data, fieldNulls[i], offsets, pool);
          break;
        default:
          VELOX_NYI();
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
  const auto numRows = data.size();
  std::vector<size_t> offsets(numRows, 0);

  return deserializeRows(rowType, data, nullptr, offsets, pool);
}

} // namespace facebook::velox::row
