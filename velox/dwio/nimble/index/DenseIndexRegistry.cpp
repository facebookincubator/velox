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
#include "velox/dwio/nimble/index/DenseIndexRegistry.h"

#include <set>

#include "velox/dwio/nimble/common/Exceptions.h"

namespace facebook::nimble::index {
namespace {

void validateIndexColumns(
    const std::vector<std::vector<std::string>>& columns) {
  NIMBLE_CHECK(!columns.empty(), "Dense index must contain columns");
  for (const auto& columnSet : columns) {
    NIMBLE_CHECK(!columnSet.empty(), "Dense index columns cannot be empty");
  }
  const std::set<std::vector<std::string>> uniqueColumns{
      columns.begin(), columns.end()};
  NIMBLE_CHECK_EQ(
      uniqueColumns.size(),
      columns.size(),
      "Dense index contains duplicate column sets");
}

} // namespace

std::unique_ptr<DenseIndexRegistry> DenseIndexRegistry::create(
    std::vector<Entry> entries,
    const IndexLookup::Options& options,
    velox::memory::MemoryPool* pool) {
  options.validate();
  if (entries.empty()) {
    return nullptr;
  }
  auto registry = std::unique_ptr<DenseIndexRegistry>(new DenseIndexRegistry());
  registry->denseIndices_.reserve(entries.size());
  std::set<std::pair<std::string, std::vector<std::string>>> indexKeys;
  for (auto& entry : entries) {
    const auto* factory = denseIndexFactory(entry.name);
    NIMBLE_CHECK_NOT_NULL(
        factory, "Unknown dense index factory: {}", entry.name);
    auto reader =
        factory->createReader(std::move(entry.section), options, pool);
    NIMBLE_CHECK_NOT_NULL(
        reader, "Dense index factory returned a null reader: {}", entry.name);
    auto columns = reader->indexColumns();
    validateIndexColumns(columns);
    for (const auto& columnSet : columns) {
      NIMBLE_CHECK(
          indexKeys.emplace(entry.name, columnSet).second,
          "Duplicate dense index '{}' columns",
          entry.name);
    }
    registry->denseIndices_.emplace_back(
        DenseIndex{std::move(entry.name), std::move(reader)});
  }
  return registry;
}

const IndexLookup* DenseIndexRegistry::findIndex(
    const std::vector<std::string>& queryColumns) const {
  for (const auto& denseIndex : denseIndices_) {
    if (const auto* index = denseIndex.reader->findIndex(queryColumns)) {
      return index;
    }
  }
  return nullptr;
}

const IndexLookup* DenseIndexRegistry::findIndex(
    std::string_view name,
    const std::vector<std::string>& queryColumns) const {
  for (const auto& denseIndex : denseIndices_) {
    if (denseIndex.name == name) {
      if (const auto* index = denseIndex.reader->findIndex(queryColumns)) {
        return index;
      }
    }
  }
  return nullptr;
}

} // namespace facebook::nimble::index
