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

#include <folly/Benchmark.h>
#include <folly/Random.h>
#include <folly/init/Init.h>

#include "velox/exec/HashTable.h"
#include "velox/exec/VectorHasher.h"
#include "velox/vector/FlatVector.h"

DEFINE_int32(batch_size, 1'024, "Number of rows per input batch");

using namespace facebook::velox;
using namespace facebook::velox::exec;

namespace {

// Key layouts that steer the adaptive hash table into a specific hash mode.
enum class KeyShape {
  // Two BIGINT keys with small ranges. Resolves to kNormalizedKey.
  kTwoBigints,
  // One BIGINT key with random 64-bit values. Resolves to kHash.
  kRandomBigint,
  // One BIGINT key and one 16-character VARCHAR key. Resolves to kHash.
  kBigintVarchar,
};

// Measures HashTable::groupProbe() with group-by tables larger than the CPU
// cache. The build case inserts every group once in random order, so nearly
// every probe row creates a new group. The probe case re-probes a built table
// in a different random order, so every probe row hits an existing group.
class GroupProbeBenchmark {
 public:
  GroupProbeBenchmark(KeyShape shape, int64_t numGroups)
      : shape_(shape),
        numGroups_(numGroups),
        pool_(memory::memoryManager()->addLeafPool()) {}

  // Returns the number of probed rows.
  uint64_t runBuild() {
    folly::BenchmarkSuspender suspender;
    auto buildBatches = makeBatches(1);
    auto table = makeTable();
    HashLookup lookup(table->hashers(), pool_.get());
    suspender.dismiss();
    probe(*table, lookup, buildBatches);
    suspender.rehire();
    checkNumDistinct(*table);
    return numGroups_;
  }

  // Returns the number of probed rows.
  uint64_t runProbe() {
    folly::BenchmarkSuspender suspender;
    auto buildBatches = makeBatches(1);
    auto probeBatches = makeBatches(2);
    auto table = makeTable();
    HashLookup lookup(table->hashers(), pool_.get());
    probe(*table, lookup, buildBatches);
    suspender.dismiss();
    probe(*table, lookup, probeBatches);
    suspender.rehire();
    checkNumDistinct(*table);
    return numGroups_;
  }

 private:
  std::vector<TypePtr> keyTypes() const {
    switch (shape_) {
      case KeyShape::kTwoBigints:
        return {BIGINT(), BIGINT()};
      case KeyShape::kRandomBigint:
        return {BIGINT()};
      case KeyShape::kBigintVarchar:
        return {BIGINT(), VARCHAR()};
    }
    VELOX_UNREACHABLE();
  }

  std::unique_ptr<BaseHashTable> makeTable() {
    std::vector<std::unique_ptr<VectorHasher>> hashers;
    const auto types = keyTypes();
    for (auto channel = 0; channel < types.size(); ++channel) {
      hashers.push_back(
          std::make_unique<VectorHasher>(types[channel], channel));
    }
    return HashTable<false>::createForAggregation(
        std::move(hashers), {}, pool_.get());
  }

  // Makes batches holding each of the 'numGroups_' keys once, in an order
  // determined by 'seed'.
  std::vector<RowVectorPtr> makeBatches(uint32_t seed) {
    std::vector<int64_t> groupIds(numGroups_);
    std::iota(groupIds.begin(), groupIds.end(), 0);
    folly::Random::DefaultGenerator rng(seed);
    std::shuffle(groupIds.begin(), groupIds.end(), rng);

    const auto types = keyTypes();
    const auto rowType = ROW(std::vector<TypePtr>(types));
    std::vector<RowVectorPtr> batches;
    for (int64_t start = 0; start < numGroups_; start += FLAGS_batch_size) {
      const auto numRows =
          std::min<int64_t>(FLAGS_batch_size, numGroups_ - start);
      std::vector<VectorPtr> children;
      for (const auto& type : types) {
        children.push_back(BaseVector::create(type, numRows, pool_.get()));
      }
      for (auto row = 0; row < numRows; ++row) {
        setKey(children, row, groupIds[start + row]);
      }
      batches.push_back(
          std::make_shared<RowVector>(
              pool_.get(), rowType, nullptr, numRows, std::move(children)));
    }
    return batches;
  }

  void setKey(std::vector<VectorPtr>& children, vector_size_t row, int64_t id) {
    auto* first = children[0]->asFlatVector<int64_t>();
    switch (shape_) {
      case KeyShape::kTwoBigints:
        first->set(row, id / 1'000);
        children[1]->asFlatVector<int64_t>()->set(row, id % 1'000);
        return;
      case KeyShape::kRandomBigint:
        // A bijective mix spreads keys over the full 64-bit range.
        first->set(row, static_cast<int64_t>(folly::hash::twang_mix64(id)));
        return;
      case KeyShape::kBigintVarchar: {
        first->set(row, id % 1'000);
        const auto name = fmt::format("item{:012d}", id / 1'000);
        children[1]->asFlatVector<StringView>()->set(row, StringView(name));
        return;
      }
    }
  }

  void probe(
      BaseHashTable& table,
      HashLookup& lookup,
      const std::vector<RowVectorPtr>& batches) {
    SelectivityVector rows;
    for (const auto& batch : batches) {
      rows.resize(batch->size());
      rows.setAll();
      table.prepareForGroupProbe(
          lookup, batch, rows, BaseHashTable::kNoSpillInputStartPartitionBit);
      table.groupProbe(lookup, BaseHashTable::kNoSpillInputStartPartitionBit);
    }
  }

  void checkNumDistinct(BaseHashTable& table) const {
    VELOX_CHECK_EQ(table.numDistinct(), numGroups_, "{}", table.toString());
  }

  const KeyShape shape_;
  const int64_t numGroups_;
  const std::shared_ptr<memory::MemoryPool> pool_;
};

void addBenchmarks(std::string_view name, KeyShape shape) {
  for (const int64_t numGroups : {1'000'000, 8'000'000, 32'000'000}) {
    const auto suffix = fmt::format("{}_{}M", name, numGroups / 1'000'000);
    folly::addBenchmark(__FILE__, "build_" + suffix, [shape, numGroups]() {
      return GroupProbeBenchmark(shape, numGroups).runBuild();
    });
    folly::addBenchmark(__FILE__, "probe_" + suffix, [shape, numGroups]() {
      return GroupProbeBenchmark(shape, numGroups).runProbe();
    });
  }
}

} // namespace

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  addBenchmarks("normalizedKey", KeyShape::kTwoBigints);
  addBenchmarks("hashBigint", KeyShape::kRandomBigint);
  addBenchmarks("hashBigintVarchar", KeyShape::kBigintVarchar);
  folly::runBenchmarks();
  return 0;
}
