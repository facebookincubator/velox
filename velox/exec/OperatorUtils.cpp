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

#include "velox/exec/OperatorUtils.h"
#include "velox/expression/EvalCtx.h"
#include "velox/vector/ConstantVector.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::exec {

void deselectRowsWithNulls(
    const RowVector& input,
    const std::vector<ChannelIndex>& channels,
    SelectivityVector& rows,
    core::ExecCtx& execCtx) {
  bool anyChange = false;
  auto numRows = input.size();
  EvalCtx evalCtx(&execCtx, nullptr, &input);
  for (auto channel : channels) {
    evalCtx.ensureFieldLoaded(channel, rows);
    auto key = input.loadedChildAt(channel);
    if (key->mayHaveNulls()) {
      auto nulls = key->flatRawNulls(rows);
      anyChange = true;
      bits::andBits(
          rows.asMutableRange().bits(),
          nulls,
          0,
          std::min(rows.size(), numRows));
    }
  }
  if (anyChange) {
    rows.updateBounds();
  }
}

uint64_t* FilterEvalCtx::getRawSelectedBits(
    vector_size_t size,
    memory::MemoryPool* pool) {
  uint64_t* rawBits;
  BaseVector::ensureBuffer<bool, uint64_t>(size, pool, &selectedBits, &rawBits);
  return rawBits;
}

vector_size_t* FilterEvalCtx::getRawSelectedIndices(
    vector_size_t size,
    memory::MemoryPool* pool) {
  vector_size_t* rawSelected;
  BaseVector::ensureBuffer<vector_size_t>(
      size, pool, &selectedIndices, &rawSelected);
  return rawSelected;
}

namespace {
vector_size_t processConstantFilterResults(
    const VectorPtr& filterResult,
    const SelectivityVector& rows) {
  auto constant = filterResult->as<ConstantVector<bool>>();
  if (constant->isNullAt(0) || constant->valueAt(0) == false) {
    return 0;
  }
  return rows.size();
}

vector_size_t processFlatFilterResults(
    const VectorPtr& filterResult,
    const SelectivityVector& rows,
    FilterEvalCtx& filterEvalCtx,
    memory::MemoryPool* pool) {
  auto size = rows.size();

  auto selectedBits = filterEvalCtx.getRawSelectedBits(size, pool);
  auto nonNullBits =
      filterResult->as<FlatVector<bool>>()->rawValues<uint64_t>();
  if (filterResult->mayHaveNulls()) {
    bits::andBits(selectedBits, nonNullBits, filterResult->rawNulls(), 0, size);
  } else {
    memcpy(selectedBits, nonNullBits, bits::nbytes(size));
  }

  vector_size_t passed = 0;
  auto* rawSelected = filterEvalCtx.getRawSelectedIndices(size, pool);
  bits::forEachSetBit(
      selectedBits, 0, size, [&rawSelected, &passed](vector_size_t row) {
        rawSelected[passed++] = row;
      });
  return passed;
}

vector_size_t processEncodedFilterResults(
    const VectorPtr& filterResult,
    const SelectivityVector& rows,
    FilterEvalCtx& filterEvalCtx,
    memory::MemoryPool* pool) {
  auto size = rows.size();

  DecodedVector& decoded = filterEvalCtx.decodedResult;
  decoded.decode(*filterResult.get(), rows);
  auto values = decoded.data<uint64_t>();
  auto nulls = decoded.nulls();
  auto indices = decoded.indices();
  auto nullIndices = decoded.nullIndices();

  vector_size_t passed = 0;
  auto* rawSelected = filterEvalCtx.getRawSelectedIndices(size, pool);
  auto* rawSelectedBits = filterEvalCtx.getRawSelectedBits(size, pool);
  memset(rawSelectedBits, 0, bits::nbytes(size));
  for (int32_t i = 0; i < size; ++i) {
    auto index = indices[i];
    auto nullIndex = nullIndices ? nullIndices[i] : i;
    if ((!nulls || !bits::isBitNull(nulls, nullIndex)) &&
        bits::isBitSet(values, index)) {
      rawSelected[passed++] = i;
      bits::setBit(rawSelectedBits, i);
    }
  }
  return passed;
}
} // namespace

vector_size_t processFilterResults(
    const VectorPtr& filterResult,
    const SelectivityVector& rows,
    FilterEvalCtx& filterEvalCtx,
    memory::MemoryPool* pool) {
  switch (filterResult->encoding()) {
    case VectorEncoding::Simple::CONSTANT:
      return processConstantFilterResults(filterResult, rows);
    case VectorEncoding::Simple::FLAT:
      return processFlatFilterResults(filterResult, rows, filterEvalCtx, pool);
    default:
      return processEncodedFilterResults(
          filterResult, rows, filterEvalCtx, pool);
  }
}

VectorPtr
wrapChild(vector_size_t size, BufferPtr mapping, const VectorPtr& child) {
  if (!mapping) {
    return child;
  }

  if (child->encoding() == VectorEncoding::Simple::CONSTANT) {
    if (size == child->size()) {
      return child;
    }
    return BaseVector::wrapInConstant(size, 0, child);
  }

  return BaseVector::wrapInDictionary(BufferPtr(nullptr), mapping, size, child);
}

RowVectorPtr
wrap(vector_size_t size, BufferPtr mapping, const RowVectorPtr& vector) {
  if (!mapping) {
    return vector;
  }

  std::vector<VectorPtr> wrappedChildren;
  wrappedChildren.reserve(vector->childrenSize());
  for (auto& child : vector->children()) {
    wrappedChildren.emplace_back(wrapChild(size, mapping, child));
  }

  return std::make_shared<RowVector>(
      vector->pool(),
      vector->type(),
      BufferPtr(nullptr),
      size,
      wrappedChildren);
}
} // namespace facebook::velox::exec
