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

#include "velox/dwio/common/SelectiveFlatMapColumnReader.h"

#include "velox/common/base/SimdUtil.h"

namespace facebook::velox::dwio::common {
namespace {

// Check if `result` can be reused to prevent allocation of a new vector. Moves
// the map values that nothing else holds to 'reusableValues', for the child
// readers to reuse.
FlatMapVector* tryReuseResult(
    const VectorPtr& result,
    const VectorPtr& distinctKeys,
    vector_size_t size,
    std::vector<VectorPtr>& reusableValues) {
  if (result.use_count() != 1) {
    return nullptr;
  }

  if (result->encoding() == VectorEncoding::Simple::FLAT_MAP) {
    auto* flatMap = static_cast<FlatMapVector*>(result.get());
    if (size > flatMap->size()) {
      for (const auto& inMap : flatMap->inMaps()) {
        if (inMap != nullptr && !inMap->unique()) {
          return nullptr;
        }
      }
    }
    reusableValues.resize(flatMap->numDistinctKeys());
    for (column_index_t i = 0; i < flatMap->numDistinctKeys(); ++i) {
      auto values = std::move(flatMap->mapValuesAt(i));
      if (values.use_count() == 1) {
        reusableValues[i] = std::move(values);
      }
    }
    if (flatMap->distinctKeys() != distinctKeys) {
      flatMap->setDistinctKeys(distinctKeys);
    }
    flatMap->resize(size);
    return flatMap;
  }
  return nullptr;
}

template <TypeKind kind>
void releaseSharedValues(BaseVector& vector) {
  using T = typename TypeTraits<kind>::NativeType;
  auto* flat = vector.asUnchecked<FlatVector<T>>();
  if (flat->values() && !flat->values()->isMutable()) {
    flat->unsafeSetValues(nullptr);
  }
}

// Drops the references of 'vector' and of its unshared descendants to buffers
// that are shared, so that the readers holding them can write them again.
void releaseSharedBuffers(BaseVector& vector) {
  switch (vector.encoding()) {
    case VectorEncoding::Simple::FLAT:
      VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH_ALL(
          releaseSharedValues, vector.typeKind(), vector);
      break;
    case VectorEncoding::Simple::ARRAY: {
      const auto& elements = vector.asUnchecked<ArrayVector>()->elements();
      if (elements.use_count() == 1) {
        releaseSharedBuffers(*elements);
      }
      break;
    }
    case VectorEncoding::Simple::MAP: {
      auto* map = vector.asUnchecked<MapVector>();
      if (map->mapKeys().use_count() == 1) {
        releaseSharedBuffers(*map->mapKeys());
      }
      if (map->mapValues().use_count() == 1) {
        releaseSharedBuffers(*map->mapValues());
      }
      break;
    }
    default:
      break;
  }
  if (vector.nulls() && !vector.nulls()->isMutable()) {
    vector.resetNulls();
  }
}
} // namespace

void SelectiveFlatMapColumnReader::getValues(
    const RowSet& rows,
    VectorPtr* result) {
  VELOX_CHECK(!scanSpec_->children().empty());
  VELOX_CHECK_NOT_NULL(
      *result, "SelectiveFlatMapColumnReader expects a non-null result");
  VELOX_CHECK(
      result->get()->type()->isMap(),
      "Struct reader expects a result of type MAP.");

  if (rows.empty()) {
    return;
  }

  auto* resultFlatMap = prepareResult(*result, keysVector_, rows.size());
  setComplexNulls(rows, *result);

  // Loop over column readers
  for (int i = 0; i < children_.size(); ++i) {
    auto& child = children_[i];
    VectorPtr values;
    if (child->allNull()) {
      values = nullValues_;
    } else if (i < reusableValues_.size()) {
      values = std::move(reusableValues_[i]);
    }
    child->getValues(rows, &values);
    if (values->isConstantEncoding() && values->isNullAt(0)) {
      nullValues_ = values;
    }
    resultFlatMap->mapValuesAt(i) = std::move(values);
    selectInMap(
        inMapBuffer(i),
        rows,
        resultFlatMap->pool(),
        resultFlatMap->inMapsAt(i, true));
  }
  reusableValues_.clear();
}

void SelectiveFlatMapColumnReader::releaseReusableResult(VectorPtr& result) {
  if (result.use_count() != 1 ||
      result->encoding() != VectorEncoding::Simple::FLAT_MAP) {
    return;
  }
  auto* flatMap = result->asUnchecked<FlatMapVector>();
  if (flatMap->nulls() && !flatMap->nulls()->isMutable()) {
    flatMap->resetNulls();
  }
  reusableValues_.resize(flatMap->numDistinctKeys());
  for (column_index_t i = 0; i < flatMap->numDistinctKeys(); ++i) {
    auto& values = flatMap->mapValuesAt(i);
    if (values.use_count() == 1) {
      releaseSharedBuffers(*values);
      reusableValues_[i] = std::move(values);
    }
  }
}

// static
void SelectiveFlatMapColumnReader::selectInMap(
    const BufferPtr& inMap,
    const RowSet& rows,
    memory::MemoryPool* pool,
    BufferPtr& selected) {
  if (inMap == nullptr) {
    selected = nullptr;
    return;
  }
  if (selected && selected->isMutable() &&
      selected->capacity() >= bits::nwords(rows.size()) * sizeof(uint64_t)) {
    selected->setSize(bits::nbytes(rows.size()));
  } else {
    selected = AlignedBuffer::allocate<bool>(rows.size(), pool);
  }
  simd::gatherBits(
      inMap->as<uint64_t>(), rows, selected->asMutable<uint64_t>());
}

FlatMapVector* SelectiveFlatMapColumnReader::prepareResult(
    VectorPtr& result,
    const VectorPtr& distinctKeys,
    vector_size_t size) {
  if (auto reused =
          tryReuseResult(result, distinctKeys, size, reusableValues_)) {
    return reused;
  }

  auto flatMap = std::make_shared<FlatMapVector>(
      result->pool(),
      result->type(),
      nullptr,
      size,
      distinctKeys,
      std::vector<VectorPtr>(distinctKeys->size()),
      std::vector<BufferPtr>{});
  // `result` may be a regular MAP vector, such as the caller's initial empty
  // result, so check that it is a FLAT_MAP before casting. If it is, it is a
  // previous batch still in use; when it has the same keys, reuse its key index
  // rather than building another one.
  if (result->encoding() == VectorEncoding::Simple::FLAT_MAP) {
    flatMap->shareKeyIndex(*result->asUnchecked<FlatMapVector>());
  }
  result = flatMap;
  return flatMap.get();
}
} // namespace facebook::velox::dwio::common
