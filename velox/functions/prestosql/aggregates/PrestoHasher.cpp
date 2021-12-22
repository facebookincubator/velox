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
#define XXH_INLINE_ALL

#include "velox/functions/prestosql/aggregates/PrestoHasher.h"
#include "velox/external/xxhash.h"
#include "velox/functions/lib/LambdaFunctionUtil.h"

namespace facebook::velox::aggregate {

namespace {

#define ENABLE_NULL_CHECK(FUNC, ...)       \
  auto hasNulls = vector_->mayHaveNulls(); \
  if (hasNulls) {                          \
    FUNC<true>(__VA_ARGS__);               \
  } else {                                 \
    FUNC<false>(__VA_ARGS__);              \
  }

template <typename T>
FOLLY_ALWAYS_INLINE int64_t hashInteger(const T& value) {
  return XXH64_round(0, value);
}

FOLLY_ALWAYS_INLINE int64_t
hashStringView(const DecodedVector& vector, vector_size_t row) {
  auto input = vector.valueAt<StringView>(row);
  return XXH64(input.data(), input.size(), 0);
}

template <typename Callable>
FOLLY_ALWAYS_INLINE void applyHashFunction(
    const SelectivityVector& rows,
    BufferPtr& hashes,
    const DecodedVector& vector,
    Callable func) {
  VELOX_CHECK_GE(hashes->size(), rows.end())
  auto rawHashes = hashes->asMutable<int64_t>();
  auto nulls = vector.mayHaveNulls();

  rows.applyToSelected([&](auto row) {
    if (nulls) {
      rawHashes[row] = vector.isNullAt(row) ? 0 : func(row);
    } else {
      rawHashes[row] = func(row);
    }
  });
}

template <typename T>
FOLLY_ALWAYS_INLINE void hashIntegral(
    const DecodedVector& vector,
    const SelectivityVector& rows,
    BufferPtr& hashes) {
  if (vector.isIdentityMapping()) {
    auto values = vector.values<int64_t>();
    applyHashFunction(rows, hashes, vector, [&](auto row) {
      return hashInteger(values[row]);
    });
  } else {
    applyHashFunction(rows, hashes, vector, [&](auto row) {
      return hashInteger(vector.valueAt<T>(row));
    });
  }
}

template <bool hasNulls>
FOLLY_ALWAYS_INLINE void hashVarchar(
    const SelectivityVector& rows,
    BufferPtr& hashes,
    DecodedVector& vector) {
  applyHashFunction(rows, hashes, vector, [&](auto row) {
    return hashStringView(vector, row);
  });
}

template <bool hasNulls>
void hashArray(
    const SelectivityVector& rows,
    BufferPtr& hashes,
    DecodedVector& vector,
    const std::vector<std::unique_ptr<PrestoHasher>>& children) {
  auto baseArray = vector.base()->as<ArrayVector>();
  auto indices = vector.indices();
  auto elementRows = functions::toElementRows(
      baseArray->elements()->size(), rows, baseArray, indices);

  BufferPtr elementHashes =
      AlignedBuffer::allocate<int64_t>(elementRows.end(), baseArray->pool());

  children[0]->hash(elementRows, elementHashes);

  auto rawSizes = baseArray->rawSizes();
  auto rawOffsets = baseArray->rawOffsets();
  auto rawNulls = baseArray->rawNulls();
  auto rawElementHashes = elementHashes->as<int64_t>();
  auto rawHashes = hashes->asMutable<int64_t>();

  rows.applyToSelected([&](auto row) {
    int64_t hash = 0;
    bool isNotNull = true;
    if constexpr (hasNulls) {
      isNotNull = !(rawNulls && bits::isBitNull(rawNulls, indices[row]));
    }
    if (isNotNull) {
      auto size = rawSizes[indices[row]];
      auto offset = rawOffsets[indices[row]];

      for (int i = 0; i < size; i++) {
        hash = 31 * hash + rawElementHashes[offset + i];
      }
    }
    rawHashes[row] = hash;
  });
}

template <bool hasNulls>
void hashMap(
    const SelectivityVector& rows,
    BufferPtr& hashes,
    DecodedVector& vector,
    const std::vector<std::unique_ptr<PrestoHasher>>& children) {
  auto baseMap = vector.base()->as<MapVector>();
  auto indices = vector.indices();
  VELOX_CHECK_EQ(children.size(), 2)

  auto elementRows = functions::toElementRows(
      baseMap->mapKeys()->size(), rows, baseMap, indices);
  BufferPtr keyHashes =
      AlignedBuffer::allocate<int64_t>(elementRows.end(), baseMap->pool());

  BufferPtr valueHashes =
      AlignedBuffer::allocate<int64_t>(elementRows.end(), baseMap->pool());

  children[0]->hash(elementRows, keyHashes);
  children[1]->hash(elementRows, valueHashes);

  auto rawKeyHashes = keyHashes->as<int64_t>();
  auto rawValueHashes = valueHashes->as<int64_t>();
  auto rawHashes = hashes->asMutable<int64_t>();

  auto rawSizes = baseMap->rawSizes();
  auto rawOffsets = baseMap->rawOffsets();
  auto rawNulls = baseMap->rawNulls();

  rows.applyToSelected([&](auto row) {
    int64_t hash = 0;
    bool isNotNull = true;
    if constexpr (hasNulls) {
      isNotNull = !(rawNulls && bits::isBitNull(rawNulls, indices[row]));
    }
    if (isNotNull) {
      auto size = rawSizes[indices[row]];
      auto offset = rawOffsets[indices[row]];

      for (int i = 0; i < size; i++) {
        hash = rawKeyHashes[offset + i] ^ rawValueHashes[offset + i];
      }
    }
    rawHashes[row] = hash;
  });
}

} // namespace

template <TypeKind kind>
FOLLY_ALWAYS_INLINE void PrestoHasher::hash(
    const SelectivityVector& rows,
    BufferPtr& hashes) {
  using T = typename TypeTraits<kind>::NativeType;
  applyHashFunction(rows, hashes, *vector_.get(), [&](auto row) {
    return hashInteger(vector_->valueAt<T>(row));
  });
}

template <>
FOLLY_ALWAYS_INLINE void PrestoHasher::hash<TypeKind::BOOLEAN>(
    const SelectivityVector& rows,
    BufferPtr& hashes) {
  applyHashFunction(rows, hashes, *vector_.get(), [&](auto row) -> int64_t {
    return vector_->valueAt<bool>(row) ? 1231 : 1237;
  });
}

template <>
FOLLY_ALWAYS_INLINE void PrestoHasher::hash<TypeKind::DATE>(
    const SelectivityVector& rows,
    BufferPtr& hashes) {
  applyHashFunction(rows, hashes, *vector_.get(), [&](auto row) {
    return hashInteger(vector_->valueAt<Date>(row).days());
  });
}

template <>
FOLLY_ALWAYS_INLINE void PrestoHasher::hash<TypeKind::REAL>(
    const SelectivityVector& rows,
    BufferPtr& hashes) {
  applyHashFunction(rows, hashes, *vector_.get(), [&](auto row) {
    return hashInteger(vector_->valueAt<int32_t>(row));
  });
}

template <>
FOLLY_ALWAYS_INLINE void PrestoHasher::hash<TypeKind::VARCHAR>(
    const SelectivityVector& rows,
    BufferPtr& hashes) {
  ENABLE_NULL_CHECK(hashVarchar, rows, hashes, *vector_.get())
}

template <>
FOLLY_ALWAYS_INLINE void PrestoHasher::hash<TypeKind::VARBINARY>(
    const SelectivityVector& rows,
    BufferPtr& hashes) {
  ENABLE_NULL_CHECK(hashVarchar, rows, hashes, *vector_.get())
}

template <>
FOLLY_ALWAYS_INLINE void PrestoHasher::hash<TypeKind::DOUBLE>(
    const SelectivityVector& rows,
    BufferPtr& hashes) {
  applyHashFunction(rows, hashes, *vector_.get(), [&](auto row) {
    return hashInteger(vector_->valueAt<int64_t>(row));
  });
}

template <>
FOLLY_ALWAYS_INLINE void PrestoHasher::hash<TypeKind::TIMESTAMP>(
    const SelectivityVector& rows,
    BufferPtr& hashes) {
  applyHashFunction(rows, hashes, *vector_.get(), [&](auto row) {
    return hashInteger((vector_->valueAt<Timestamp>(row)).toMillis());
  });
}

template <>
void PrestoHasher::hash<TypeKind::ARRAY>(
    const SelectivityVector& rows,
    BufferPtr& hashes) {
  ENABLE_NULL_CHECK(hashArray, rows, hashes, *vector_.get(), children_)
}

template <>
void PrestoHasher::hash<TypeKind::MAP>(
    const SelectivityVector& rows,
    BufferPtr& hashes) {
  ENABLE_NULL_CHECK(hashMap, rows, hashes, *vector_.get(), children_)
}

template <>
void PrestoHasher::hash<TypeKind::ROW>(
    const SelectivityVector& rows,
    BufferPtr& hashes) {
  auto baseRow = vector_->base()->as<RowVector>();
  auto indices = vector_->indices();
  SelectivityVector elementRows;

  if (vector_->isIdentityMapping()) {
    elementRows = rows;
  } else {
    elementRows = SelectivityVector(baseRow->size(), false);
    rows.applyToSelected(
        [&](auto row) { elementRows.setValid(indices[row], true); });
    elementRows.updateBounds();
  }

  BufferPtr childHashes =
      AlignedBuffer::allocate<int64_t>(elementRows.end(), baseRow->pool());

  auto rawHashes = hashes->asMutable<int64_t>();
  auto rowChildHashes = childHashes->as<int64_t>();

  std::fill_n(rawHashes, rows.end(), 1);

  for (int i = 0; i < baseRow->childrenSize(); i++) {
    children_[i]->hash(elementRows, childHashes);

    rows.applyToSelected([&](auto row) {
      rawHashes[row] = 31 * rawHashes[row] + rowChildHashes[indices[row]];
    });
  }
}

// BufferPtr is modified and results of hash are stored there.
void PrestoHasher::hash(const SelectivityVector& rows, BufferPtr& hashes) {
  auto kind = vector_->base()->typeKind();
  VELOX_DYNAMIC_TYPE_DISPATCH(hash, kind, rows, hashes);
}

void PrestoHasher::createChildren() {
  auto kind = vector_->base()->typeKind();
  if (kind == TypeKind::ARRAY) {
    auto baseArray = vector_->base()->as<ArrayVector>();
    SelectivityVector elementRows(baseArray->elements()->size());
    children_.push_back(
        std::make_unique<PrestoHasher>(*baseArray->elements(), elementRows));
  } else if (kind == TypeKind::MAP) {
    auto baseMap = vector_->base()->as<MapVector>();
    SelectivityVector elementRows(baseMap->mapKeys()->size());
    // Decode key
    children_.push_back(
        std::make_unique<PrestoHasher>(*baseMap->mapKeys(), elementRows));
    // Decode values
    children_.push_back(
        std::make_unique<PrestoHasher>(*baseMap->mapValues(), elementRows));

  } else if (kind == TypeKind::ROW) {
    auto baseRow = vector_->base()->as<RowVector>();

    children_.reserve(baseRow->childrenSize());
    SelectivityVector rows(vector_->size());
    for (int i = 0; i < baseRow->childrenSize(); i++) {
      children_.push_back(
          std::make_unique<PrestoHasher>(*baseRow->childAt(i), rows));
    }
  }
}

} // namespace facebook::velox::aggregate