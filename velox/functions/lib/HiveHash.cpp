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
#include "velox/functions/lib/HiveHash.h"

#include "velox/vector/ComplexVector.h"

namespace facebook::velox::functions {

template <TypeKind kind>
void HiveHash::hashTyped(
    const DecodedVector& values,
    const SelectivityVector& rows,
    bool mix,
    std::vector<uint32_t>& hashes,
    size_t /* poolIndex */) {
  if constexpr (
      TypeTraits<kind>::isPrimitiveType && kind != TypeKind::HUGEINT) {
    hashPrimitive<kind>(values, rows, mix, hashes);
  } else {
    VELOX_UNSUPPORTED(
        "Hive hash doesn't support type: {}", TypeTraits<kind>::name);
  }
}

template <>
void HiveHash::hashTyped<TypeKind::ARRAY>(
    const DecodedVector& values,
    const SelectivityVector& rows,
    bool mix,
    std::vector<uint32_t>& hashes,
    size_t poolIndex) {
  auto& elementsDecoded = getDecodedVector(poolIndex);
  auto& elementsRows = getRows(poolIndex);
  auto& elementsHashes = getHashes(poolIndex);

  const auto* arrayVector = values.base()->as<ArrayVector>();
  const vector_size_t numElements = arrayVector->elements()->size();
  elementsRows.resizeFill(numElements, false);
  elementsHashes.resize(numElements);

  rows.applyToSelected([&](auto row) {
    if (!values.isNullAt(row)) {
      const auto index = values.index(row);
      const auto offset = arrayVector->offsetAt(index);
      const auto length = arrayVector->sizeAt(index);

      elementsRows.setValidRange(offset, offset + length, true);
    }
  });

  elementsRows.updateBounds();

  elementsDecoded.decode(*arrayVector->elements(), elementsRows);

  hash(elementsDecoded, elementsRows, false, elementsHashes, poolIndex + 1);

  rows.applyToSelected([&](auto row) {
    uint32_t hash{0};

    if (!values.isNullAt(row)) {
      const auto index = values.index(row);
      const auto offset = arrayVector->offsetAt(index);
      const auto length = arrayVector->sizeAt(index);

      for (vector_size_t i = offset; i < offset + length; ++i) {
        HiveHash::mergeHash(true, elementsHashes[i], hash);
      }
    }

    HiveHash::mergeHash(mix, hash, hashes[row]);
  });
}

template <>
void HiveHash::hashTyped<TypeKind::MAP>(
    const DecodedVector& values,
    const SelectivityVector& rows,
    bool mix,
    std::vector<uint32_t>& hashes,
    size_t poolIndex) {
  auto& valuesDecoded = getDecodedVector(poolIndex);
  auto& keysDecoded = getDecodedVector(poolIndex + 1);
  auto& elementsRows = getRows(poolIndex);
  auto& valuesHashes = getHashes(poolIndex);
  auto& keysHashes = getHashes(poolIndex + 1);

  const auto* mapVector = values.base()->as<MapVector>();
  const vector_size_t numElements = mapVector->mapKeys()->size();
  elementsRows.resizeFill(numElements, false);
  keysHashes.resize(numElements);
  valuesHashes.resize(numElements);

  rows.applyToSelected([&](auto row) {
    if (!values.isNullAt(row)) {
      const auto index = values.index(row);
      const auto offset = mapVector->offsetAt(index);
      const auto length = mapVector->sizeAt(index);

      elementsRows.setValidRange(offset, offset + length, true);
    }
  });

  elementsRows.updateBounds();

  keysDecoded.decode(*mapVector->mapKeys(), elementsRows);
  valuesDecoded.decode(*mapVector->mapValues(), elementsRows);

  hash(keysDecoded, elementsRows, false, keysHashes, poolIndex + 2);

  hash(valuesDecoded, elementsRows, false, valuesHashes, poolIndex + 2);

  rows.applyToSelected([&](auto row) {
    uint32_t hash{0};

    if (!values.isNullAt(row)) {
      const auto index = values.index(row);
      const auto offset = mapVector->offsetAt(index);
      const auto length = mapVector->sizeAt(index);

      for (vector_size_t i = offset; i < offset + length; ++i) {
        hash += keysHashes[i] ^ valuesHashes[i];
      }
    }

    HiveHash::mergeHash(mix, hash, hashes[row]);
  });
}

template <>
void HiveHash::hashTyped<TypeKind::ROW>(
    const DecodedVector& values,
    const SelectivityVector& rows,
    bool mix,
    std::vector<uint32_t>& hashes,
    size_t poolIndex) {
  const auto* rowVector = values.base()->as<RowVector>();
  if (rowVector->childrenSize() == 0) {
    rows.applyToSelected(
        [&](auto row) { HiveHash::mergeHash(mix, 0, hashes[row]); });
    return;
  }

  auto& childDecodedVector = getDecodedVector(poolIndex);
  auto& childRows = getRows(poolIndex);
  auto& childHashes = getHashes(poolIndex);

  childRows.resizeFill(rowVector->size(), false);
  childHashes.resize(rowVector->size());

  rows.applyToSelected([&](auto row) {
    if (!values.isNullAt(row)) {
      childRows.setValid(values.index(row), true);
    }
  });

  childRows.updateBounds();

  for (vector_size_t i = 0; i < rowVector->childrenSize(); ++i) {
    auto& child = rowVector->childAt(i);
    childDecodedVector.decode(*child, childRows);
    hash(childDecodedVector, childRows, i > 0, childHashes, poolIndex + 1);
  }

  rows.applyToSelected([&](auto row) {
    HiveHash::mergeHash(
        mix,
        values.isNullAt(row) ? 0 : childHashes[values.index(row)],
        hashes[row]);
  });
}

void HiveHash::hash(
    const DecodedVector& values,
    const SelectivityVector& rows,
    bool mix,
    std::vector<uint32_t>& hashes,
    size_t poolIndex) {
  VELOX_DYNAMIC_TYPE_DISPATCH_ALL(
      hashTyped,
      values.base()->typeKind(),
      values,
      rows,
      mix,
      hashes,
      poolIndex);
}

DecodedVector& HiveHash::getDecodedVector(size_t poolIndex) {
  while (poolIndex >= decodedVectorsPool_.size()) {
    decodedVectorsPool_.push_back(std::make_unique<DecodedVector>());
  }

  return *decodedVectorsPool_[poolIndex];
}

SelectivityVector& HiveHash::getRows(size_t poolIndex) {
  while (poolIndex >= rowsPool_.size()) {
    rowsPool_.push_back(std::make_unique<SelectivityVector>());
  }

  return *rowsPool_[poolIndex];
}

std::vector<uint32_t>& HiveHash::getHashes(size_t poolIndex) {
  while (poolIndex >= hashesPool_.size()) {
    hashesPool_.push_back(std::make_unique<std::vector<uint32_t>>());
  }

  return *hashesPool_[poolIndex];
}

} // namespace facebook::velox::functions
