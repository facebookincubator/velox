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

#include "OrderByBenchmarkUtil.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

using namespace facebook::velox;

namespace {
RowTypePtr rowWithName(std::vector<TypePtr>&& types) {
  std::vector<std::string> names;
  for (auto i = 0; i < types.size(); ++i) {
    names.emplace_back(fmt::format("c{}", i));
  }
  return ROW(std::move(names), std::move(types));
}
} // namespace

namespace facebook::velox {

std::vector<RowTypePtr> OrderByBenchmarkUtil::bigintRowTypes(bool noPayload) {
  if (noPayload) {
    return {
        rowWithName({BIGINT()}),
        rowWithName({BIGINT(), BIGINT()}),
        rowWithName({BIGINT(), BIGINT(), BIGINT()}),
        rowWithName({BIGINT(), BIGINT(), BIGINT(), BIGINT()}),
    };
  } else {
    return {
        rowWithName({BIGINT(), VARCHAR(), VARCHAR()}),
        rowWithName({BIGINT(), BIGINT(), VARCHAR(), VARCHAR()}),
        rowWithName({BIGINT(), BIGINT(), BIGINT(), VARCHAR(), VARCHAR()}),
        rowWithName(
            {BIGINT(), BIGINT(), BIGINT(), BIGINT(), VARCHAR(), VARCHAR()}),
    };
  }
}

std::vector<RowTypePtr> OrderByBenchmarkUtil::largeVarcharRowTypes() {
  return {
      rowWithName({VARCHAR()}),
      rowWithName({VARCHAR(), VARCHAR()}),
      rowWithName({VARCHAR(), VARCHAR(), VARCHAR()}),
      rowWithName({VARCHAR(), VARCHAR(), VARCHAR(), VARCHAR()}),
  };
}

RowVectorPtr OrderByBenchmarkUtil::fuzzRows(
    const RowTypePtr& rowType,
    size_t numRows,
    int numKeys,
    memory::MemoryPool* pool) {
  VectorFuzzer fuzzer({.vectorSize = numRows}, pool);
  VectorFuzzer fuzzerWithNulls({.vectorSize = numRows, .nullRatio = 0.7}, pool);
  std::vector<VectorPtr> children;

  // Fuzz keys: for front keys (column 0 to numKeys -2) use high
  // nullRatio to enforce all columns to be compared.
  {
    for (auto i = 0; i < numKeys - 1; ++i) {
      children.push_back(fuzzerWithNulls.fuzz(rowType->childAt(i)));
    }
    children.push_back(fuzzer.fuzz(rowType->childAt(numKeys - 1)));
  }
  // Fuzz payload
  {
    for (auto i = numKeys; i < rowType->size(); ++i) {
      children.push_back(fuzzer.fuzz(rowType->childAt(i)));
    }
  }
  return std::make_shared<RowVector>(
      pool, rowType, nullptr, numRows, std::move(children));
}
} // namespace facebook::velox