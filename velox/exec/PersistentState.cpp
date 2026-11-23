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
#include "velox/exec/PersistentState.h"

namespace facebook::velox::exec {

namespace {

// Deep-copies 'source' into 'pool' so the result stays valid after the source's
// owning task (and its pool) is destroyed.
RowVectorPtr copyInto(const RowVectorPtr& source, memory::MemoryPool* pool) {
  auto target =
      BaseVector::create<RowVector>(source->type(), source->size(), pool);
  target->copy(source.get(), 0, 0, source->size());
  return target;
}

// Deep-copies the non-empty 'batches' into 'pool', dropping null/empty ones.
// A batch already allocated in 'pool' is taken as is: the copy exists only to
// outlive the pool the batch came from, which is nothing to do when that pool
// is this one (the caller captured the rows here directly).
std::vector<RowVectorPtr> copyBatches(
    const std::vector<RowVectorPtr>& batches,
    memory::MemoryPool* pool) {
  std::vector<RowVectorPtr> copies;
  copies.reserve(batches.size());
  for (const auto& batch : batches) {
    if (batch == nullptr || batch->size() == 0) {
      continue;
    }
    if (batch->pool() == pool) {
      copies.push_back(batch);
    } else {
      copies.push_back(copyInto(batch, pool));
    }
  }
  return copies;
}

} // namespace

void PersistentState::declareVector(std::string_view name, bool append) {
  vectors_.withWLock(
      [&](auto& map) { map[std::string{name}].append = append; });
}

void PersistentState::setVector(
    std::string_view name,
    const std::vector<RowVectorPtr>& batches) {
  auto copies = copyBatches(batches, pool_.get());
  vectors_.withWLock([&](auto& map) {
    auto& entry = map[std::string{name}];
    // Seed the frontier too, so an append entry's first in-loop read returns
    // the seed (e.g. a recursive CTE's base case).
    if (entry.append) {
      entry.frontier = copies;
    }
    entry.batches = std::move(copies);
  });
}

void PersistentState::writeVector(
    std::string_view name,
    const std::vector<RowVectorPtr>& batches) {
  auto copies = copyBatches(batches, pool_.get());
  vectors_.withWLock([&](auto& map) {
    auto& entry = map[std::string{name}];
    if (!entry.append) {
      entry.batches = std::move(copies);
      return;
    }
    // The new rows become the frontier (the delta the next in-loop read
    // returns -- empty when nothing was produced, which signals convergence)
    // and also accumulate into the full contents.
    entry.frontier = copies;
    entry.batches.insert(
        entry.batches.end(),
        std::make_move_iterator(copies.begin()),
        std::make_move_iterator(copies.end()));
  });
}

std::vector<RowVectorPtr> PersistentState::getVector(
    std::string_view name) const {
  return vectors_.withRLock([&](const auto& map) -> std::vector<RowVectorPtr> {
    auto it = map.find(name);
    if (it == map.end()) {
      return {};
    }
    return it->second.batches;
  });
}

std::vector<RowVectorPtr> PersistentState::readVector(
    std::string_view name) const {
  return vectors_.withRLock([&](const auto& map) -> std::vector<RowVectorPtr> {
    auto it = map.find(name);
    if (it == map.end()) {
      return {};
    }
    return it->second.append ? it->second.frontier : it->second.batches;
  });
}

vector_size_t PersistentState::numRows(std::string_view name) const {
  return vectors_.withRLock([&](const auto& map) -> vector_size_t {
    auto it = map.find(name);
    if (it == map.end()) {
      return 0;
    }
    vector_size_t total{0};
    for (const auto& batch : it->second.batches) {
      total += batch->size();
    }
    return total;
  });
}

void PersistentState::setHashTable(
    std::string_view name,
    HashTableEntry entry) {
  hashTables_.withWLock(
      [&](auto& map) { map[std::string{name}] = std::move(entry); });
}

std::optional<HashTableEntry> PersistentState::getHashTable(
    std::string_view name) const {
  return hashTables_.withRLock(
      [&](const auto& map) -> std::optional<HashTableEntry> {
        auto it = map.find(name);
        if (it == map.end()) {
          return std::nullopt;
        }
        return it->second;
      });
}

bool PersistentState::hasVector(std::string_view name) const {
  return vectors_.withRLock(
      [&](const auto& map) { return map.count(name) > 0; });
}

bool PersistentState::hasHashTable(std::string_view name) const {
  return hashTables_.withRLock(
      [&](const auto& map) { return map.count(name) > 0; });
}

} // namespace facebook::velox::exec
