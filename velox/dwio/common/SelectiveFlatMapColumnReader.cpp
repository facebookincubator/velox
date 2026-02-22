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

#include "velox/dwio/common/SelectiveFlatMapColumnReader.h"

namespace facebook::velox::dwio::common {
namespace {

// Check if `result` can be reused to prevent allocation of a new vector.
FlatMapVector* tryReuseResult(
    const VectorPtr& result,
    const VectorPtr& distinctKeys,
    vector_size_t size) {
  if (result.use_count() != 1) {
    return nullptr;
  }

  if (result->encoding() == VectorEncoding::Simple::FLAT_MAP) {
    auto* flatMap = static_cast<FlatMapVector*>(result.get());
    if (flatMap->distinctKeys() != distinctKeys) {
      flatMap->setDistinctKeys(distinctKeys);
    }
    flatMap->resize(size);
    return flatMap;
  }
  return nullptr;
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
    child->getValues(rows, &values);
    resultFlatMap->mapValuesAt(i) = values;

    const auto& inMap = inMapBuffer(i);
    if (inMap) {
      resultFlatMap->inMapsAt(i, true) = inMap;
    }
  }
}

FlatMapVector* SelectiveFlatMapColumnReader::prepareResult(
    VectorPtr& result,
    const VectorPtr& distinctKeys,
    vector_size_t size) const {
  if (auto reused = tryReuseResult(result, distinctKeys, size)) {
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
  result = flatMap;
  return flatMap.get();
}
} // namespace facebook::velox::dwio::common
