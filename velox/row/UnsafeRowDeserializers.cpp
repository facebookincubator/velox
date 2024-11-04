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
#include "velox/row/UnsafeRowDeserializers.h"

namespace facebook::velox::row {
namespace {
// Returns the offset of a column to starting memory address of one row.
// @param nullBitsetWidthInBytes The null-tracking bit set is aligned to 8-byte
// word boundaries. It stores one bit per field.
// @param columnIdx column index.
inline int64_t getFieldOffset(
    int64_t nullBitsetWidthInBytes,
    column_index_t columnIdx) {
  return nullBitsetWidthInBytes + UnsafeRow::kFieldWidthBytes * columnIdx;
}

inline bool isNullAt(const uint8_t* memoryAddress, vector_size_t row) {
  return bits::isBitSet(memoryAddress, row);
}

size_t getTotalStringSize(
    column_index_t columnIdx,
    vector_size_t numRows,
    int64_t fieldOffset,
    const std::vector<int64_t>& offsets,
    const uint8_t* memoryAddress) {
  size_t size = 0;
  for (auto row = 0; row < numRows; row++) {
    if (isNullAt(memoryAddress + offsets[row], columnIdx)) {
      continue;
    }

    int64_t offsetAndSize =
        *(int64_t*)(memoryAddress + offsets[row] + fieldOffset);
    int32_t length = static_cast<int32_t>(offsetAndSize);
    if (!StringView::isInline(length)) {
      size += length;
    }
  }
  return size;
}

template <TypeKind Kind>
VectorPtr createFlatVectorFast(
    const TypePtr& type,
    column_index_t columnIdx,
    vector_size_t numRows,
    int64_t fieldOffset,
    const std::vector<int64_t>& offsets,
    const uint8_t* memoryAddress,
    memory::MemoryPool* pool) {
  using T = typename TypeTraits<Kind>::NativeType;
  constexpr uint32_t typeWidth = sizeof(T);
  auto column = BaseVector::create<FlatVector<T>>(type, numRows, pool);
  auto rawValues = column->template mutableRawValues<uint8_t>();
  const auto shift = __builtin_ctz(typeWidth);
  for (auto row = 0; row < numRows; row++) {
    if (!isNullAt(memoryAddress + offsets[row], columnIdx)) {
      const uint8_t* srcPtr = (memoryAddress + offsets[row] + fieldOffset);
      uint8_t* destPtr = rawValues + (row << shift);
      memcpy(destPtr, srcPtr, typeWidth);
    } else {
      column->setNull(row, true);
    }
  }
  return column;
}

template <>
VectorPtr createFlatVectorFast<TypeKind::HUGEINT>(
    const TypePtr& type,
    column_index_t columnIdx,
    vector_size_t numRows,
    int64_t fieldOffset,
    const std::vector<int64_t>& offsets,
    const uint8_t* memoryAddress,
    memory::MemoryPool* pool) {
  auto column = BaseVector::create<FlatVector<int128_t>>(type, numRows, pool);
  auto rawValues = column->mutableRawValues<uint8_t>();
  for (auto row = 0; row < numRows; row++) {
    if (!isNullAt(memoryAddress + offsets[row], columnIdx)) {
      const int64_t offsetAndSize =
          *(int64_t*)(memoryAddress + offsets[row] + fieldOffset);
      const int32_t length = static_cast<int32_t>(offsetAndSize);
      const int32_t wordOffset = static_cast<int32_t>(offsetAndSize >> 32);
      rawValues[row] =
          UnsafeRowPrimitiveBatchDeserializer::deserializeLongDecimal(
              std::string_view(reinterpret_cast<const char*>(
                  memoryAddress + offsets[row] + wordOffset, length)));
    } else {
      column->setNull(row, true);
    }
  }
  return column;
}

template <>
VectorPtr createFlatVectorFast<TypeKind::BOOLEAN>(
    const TypePtr& type,
    column_index_t columnIdx,
    vector_size_t numRows,
    int64_t fieldOffset,
    const std::vector<int64_t>& offsets,
    const uint8_t* memoryAddress,
    memory::MemoryPool* pool) {
  auto column = BaseVector::create<FlatVector<bool>>(type, numRows, pool);
  auto rawValues = column->mutableRawValues<uint64_t>();
  for (auto row = 0; row < numRows; row++) {
    if (!isNullAt(memoryAddress + offsets[row], columnIdx)) {
      const bool value = *(bool*)(memoryAddress + offsets[row] + fieldOffset);
      bits::setBit(rawValues, row, value);
    } else {
      column->setNull(row, true);
    }
  }
  return column;
}

template <>
VectorPtr createFlatVectorFast<TypeKind::TIMESTAMP>(
    const TypePtr& type,
    column_index_t columnIdx,
    vector_size_t numRows,
    int64_t fieldOffset,
    const std::vector<int64_t>& offsets,
    const uint8_t* memoryAddress,
    memory::MemoryPool* pool) {
  auto column = BaseVector::create<FlatVector<Timestamp>>(type, numRows, pool);
  for (auto row = 0; row < numRows; row++) {
    if (!isNullAt(memoryAddress + offsets[row], columnIdx)) {
      const int64_t value =
          *(int64_t*)(memoryAddress + offsets[row] + fieldOffset);
      column->set(row, Timestamp::fromMicros(value));
    } else {
      column->setNull(row, true);
    }
  }
  return column;
}

template <>
VectorPtr createFlatVectorFast<TypeKind::VARCHAR>(
    const TypePtr& type,
    column_index_t columnIdx,
    vector_size_t numRows,
    int64_t fieldOffset,
    const std::vector<int64_t>& offsets,
    const uint8_t* memoryAddress,
    memory::MemoryPool* pool) {
  auto column = BaseVector::create<FlatVector<StringView>>(type, numRows, pool);
  const auto totalSize = getTotalStringSize(
      columnIdx, numRows, fieldOffset, offsets, memoryAddress);
  char* rawBuffer = column->getRawStringBufferWithSpace(totalSize, true);
  for (auto row = 0; row < numRows; row++) {
    if (!isNullAt(memoryAddress + offsets[row], columnIdx)) {
      const int64_t offsetAndSize =
          *(int64_t*)(memoryAddress + offsets[row] + fieldOffset);
      const int32_t length = static_cast<int32_t>(offsetAndSize);
      const int32_t wordOffset = static_cast<int32_t>(offsetAndSize >> 32);
      auto* valueSrc = memoryAddress + offsets[row] + wordOffset;
      if (StringView::isInline(length)) {
        column->set(
            row, StringView(reinterpret_cast<const char*>(valueSrc), length));
      } else {
        memcpy(rawBuffer, valueSrc, length);
        column->setNoCopy(row, StringView(rawBuffer, length));
        rawBuffer += length;
      }
    } else {
      column->setNull(row, true);
    }
  }
  return column;
}

template <>
VectorPtr createFlatVectorFast<TypeKind::VARBINARY>(
    const TypePtr& type,
    column_index_t columnIdx,
    vector_size_t numRows,
    int64_t fieldOffset,
    const std::vector<int64_t>& offsets,
    const uint8_t* memoryAddress,
    memory::MemoryPool* pool) {
  return createFlatVectorFast<TypeKind::VARCHAR>(
      type, columnIdx, numRows, fieldOffset, offsets, memoryAddress, pool);
}

VectorPtr createUnknownFlatVector(
    vector_size_t numRows,
    memory::MemoryPool* pool) {
  const auto nulls = allocateNulls(numRows, pool, bits::kNull);
  return std::make_shared<FlatVector<UnknownValue>>(
      pool,
      UNKNOWN(),
      nulls,
      numRows,
      nullptr, // values
      std::vector<BufferPtr>{}); // stringBuffers
}

bool fastSupported(const RowTypePtr& type) {
  for (auto i = 0; i < type->size(); i++) {
    const auto kind = type->childAt(i)->kind();
    switch (kind) {
      case TypeKind::ARRAY:
      case TypeKind::MAP:
      case TypeKind::ROW:
        return false;
      default:
        break;
    }
  }
  return true;
}

VectorPtr deserializeFast(
    const uint8_t* memoryAddress,
    const RowTypePtr& type,
    const std::vector<int64_t>& offsets,
    vector_size_t numRows,
    memory::MemoryPool* pool) {
  const auto numFields = type->size();
  const int64_t nullBitsetWidthInBytes = UnsafeRow::getNullLength(numFields);
  std::vector<VectorPtr> columns(numFields);
  for (auto i = 0; i < numFields; i++) {
    const auto fieldOffset = getFieldOffset(nullBitsetWidthInBytes, i);
    const auto& colType = type->childAt(i);
    if (colType->kind() == TypeKind::UNKNOWN) {
      columns[i] = createUnknownFlatVector(numRows, pool);
    } else {
      columns[i] = VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
          createFlatVectorFast,
          colType->kind(),
          colType,
          i,
          numRows,
          fieldOffset,
          offsets,
          memoryAddress,
          pool);
    }
  }

  return std::make_shared<RowVector>(
      pool, type, BufferPtr(nullptr), numRows, std::move(columns));
}
} // namespace

// static
VectorPtr UnsafeRowDeserializer::deserialize(
    const uint8_t* memoryAddress,
    const RowTypePtr& type,
    const std::vector<int64_t>& offsets,
    memory::MemoryPool* pool) {
  const vector_size_t numRows = offsets.size() - 1;
  if (fastSupported(type)) {
    return deserializeFast(memoryAddress, type, offsets, numRows, pool);
  }
  std::vector<std::optional<std::string_view>> data;
  for (auto i = 0; i < numRows; i++) {
    const auto length = offsets[i + 1] - offsets[i];
    data.emplace_back(std::string_view(
        reinterpret_cast<const char*>(memoryAddress + offsets[i]), length));
  }
  return deserialize(data, type, pool);
}
} // namespace facebook::velox::row
