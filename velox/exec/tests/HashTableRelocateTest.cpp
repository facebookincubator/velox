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

#include <gtest/gtest.h>

#include "velox/exec/HashTable.h"
#include "velox/exec/VectorHasher.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;

namespace facebook::velox::exec::test {
namespace {

// Drives a stock HashTable through groupProbe to exercise relocatePayload: the
// destination ("tier") container lives on an ordinary second memory pool, which
// is all the bucket-repoint seam needs -- no real far-memory device.
class HashTableRelocateTest : public testing::Test,
                              public velox::test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  void SetUp() override {
    tierPool_ =
        memory::memoryManager()->addLeafPool("hash-table-relocate-tier");
  }

  // Builds a single-bigint-key aggregation table with no accumulators.
  std::unique_ptr<HashTable<false>> makeTable() {
    std::vector<std::unique_ptr<VectorHasher>> hashers;
    hashers.push_back(VectorHasher::create(BIGINT(), 0));
    return HashTable<false>::createForAggregation(
        std::move(hashers), {}, pool_.get());
  }

  // Probes 'keys'; hits and newGroups land in 'lookup'.
  void
  probe(HashTable<false>& table, HashLookup& lookup, const VectorPtr& keys) {
    auto input = makeRowVector({keys});
    SelectivityVector rows(keys->size());
    table.prepareForGroupProbe(
        lookup, input, rows, BaseHashTable::kNoSpillInputStartPartitionBit);
    table.groupProbe(lookup, BaseHashTable::kNoSpillInputStartPartitionBit);
  }

  // More distinct keys than VectorHasher::kMaxDistinct, spread wide, so the
  // hasher cannot assign value ids and the table uses a bucketed hash mode.
  static constexpr vector_size_t kBucketedSize = 150'000;

  VectorPtr wideKeys(vector_size_t size, int64_t base = 0) {
    return makeFlatVector<int64_t>(
        size, [base](auto row) { return (row + base) * 1'000'003LL + 7; });
  }

  // Extracts the key column of 'row' from 'container'.
  int64_t keyAt(RowContainer* container, char* row) {
    auto result = BaseVector::create(BIGINT(), 1, pool_.get());
    container->extractColumn(&row, 1, 0, result);
    return result->asFlatVector<int64_t>()->valueAt(0);
  }

  std::shared_ptr<memory::MemoryPool> tierPool_;
};

TEST_F(HashTableRelocateTest, relocatePayloadAndReprobe) {
  constexpr vector_size_t kSize = kBucketedSize;
  auto table = makeTable();
  HashLookup lookup(table->hashers(), pool_.get());
  auto keys = wideKeys(kSize);

  probe(*table, lookup, keys);
  ASSERT_EQ(lookup.newGroups.size(), kSize);
  ASSERT_NE(table->hashMode(), BaseHashTable::HashMode::kArray);

  auto* dest = table->relocatePayload(tierPool_.get());
  EXPECT_EQ(table->rows()->numRows(), 0);
  EXPECT_EQ(dest->numRows(), kSize);
  EXPECT_EQ(table->numDistinct(), kSize);
  EXPECT_EQ(table->numRowContainers(), 2);

  // Every key resolves to its relocated row, creating no new groups.
  probe(*table, lookup, keys);
  EXPECT_TRUE(lookup.newGroups.empty());
  for (vector_size_t i = 0; i < kSize; i += 1'000) {
    EXPECT_EQ(
        keyAt(dest, lookup.hits[i]), keys->asFlatVector<int64_t>()->valueAt(i));
  }
}

TEST_F(HashTableRelocateTest, relocateInArrayMode) {
  constexpr vector_size_t kSize = 1'000;
  auto table = makeTable();
  HashLookup lookup(table->hashers(), pool_.get());
  // A dense small range keeps the table in kArray mode.
  auto keys = makeFlatVector<int64_t>(kSize, [](auto row) { return row; });

  probe(*table, lookup, keys);
  ASSERT_EQ(table->hashMode(), BaseHashTable::HashMode::kArray);
  auto* dest = table->relocatePayload(tierPool_.get());
  EXPECT_EQ(dest->numRows(), kSize);

  probe(*table, lookup, keys);
  EXPECT_TRUE(lookup.newGroups.empty());
  constexpr vector_size_t kRow = 42;
  EXPECT_EQ(
      keyAt(dest, lookup.hits[kRow]),
      keys->asFlatVector<int64_t>()->valueAt(kRow));
}

TEST_F(HashTableRelocateTest, relocateSurvivesRehash) {
  constexpr vector_size_t kSize = 1'000;
  auto table = makeTable();
  HashLookup lookup(table->hashers(), pool_.get());
  auto keys = wideKeys(kSize);

  probe(*table, lookup, keys);
  table->relocatePayload(tierPool_.get());

  // Grow well past capacity to force a rehash. The rebuild lists rows from both
  // containers and reindexes the relocated rows from the tier container.
  constexpr vector_size_t kGrow = 100'000;
  auto growKeys = makeFlatVector<int64_t>(
      kGrow, [](auto row) { return -(row * 999'983LL + 11); });
  probe(*table, lookup, growKeys);
  EXPECT_EQ(lookup.newGroups.size(), kGrow);

  probe(*table, lookup, keys);
  EXPECT_TRUE(lookup.newGroups.empty());
  EXPECT_EQ(table->numDistinct(), kSize + kGrow);
}

TEST_F(HashTableRelocateTest, repeatedRelocationAddsContainers) {
  constexpr vector_size_t kFirst = 50'000;
  constexpr vector_size_t kSecond = 50'000;
  auto table = makeTable();
  HashLookup lookup(table->hashers(), pool_.get());

  // First batch relocates into a freshly-created tier container.
  auto first = wideKeys(kFirst);
  probe(*table, lookup, first);
  auto* dest = table->relocatePayload(tierPool_.get());
  EXPECT_EQ(dest->numRows(), kFirst);
  EXPECT_EQ(table->numRowContainers(), 2);

  // New groups after a relocation land in DRAM 'rows_'.
  auto second = wideKeys(kSecond, /*base=*/kFirst);
  probe(*table, lookup, second);
  EXPECT_EQ(lookup.newGroups.size(), kSecond);
  EXPECT_EQ(table->rows()->numRows(), kSecond);

  // A second relocation clones the new DRAM rows into a fresh tier container:
  // run cloning leaves the source empty, so it cannot append into the prior
  // (partially filled) container.
  auto* dest2 = table->relocatePayload(tierPool_.get());
  EXPECT_NE(dest2, dest);
  EXPECT_EQ(table->numRowContainers(), 3);
  EXPECT_EQ(table->rows()->numRows(), 0);
  EXPECT_EQ(dest->numRows(), kFirst);
  EXPECT_EQ(dest2->numRows(), kSecond);
  EXPECT_EQ(table->numDistinct(), kFirst + kSecond);

  // Both batches resolve to their relocated rows, creating no new groups.
  probe(*table, lookup, first);
  EXPECT_TRUE(lookup.newGroups.empty());
  probe(*table, lookup, second);
  EXPECT_TRUE(lookup.newGroups.empty());
}

// A payload large enough to spill the bump allocator past the huge-page
// threshold spans multiple allocation runs (small runs, then huge-page runs).
// Relocation must reproduce every run's size and content; if a non-last run's
// size were mis-cloned, listRows would mis-read rows and the reprobe would
// create new groups.
TEST_F(HashTableRelocateTest, relocateAcrossManyRuns) {
  constexpr vector_size_t kSize = 400'000;
  auto table = makeTable();
  HashLookup lookup(table->hashers(), pool_.get());
  auto keys = wideKeys(kSize);

  probe(*table, lookup, keys);
  ASSERT_EQ(lookup.newGroups.size(), kSize);

  auto* dest = table->relocatePayload(tierPool_.get());
  EXPECT_EQ(table->rows()->numRows(), 0);
  EXPECT_EQ(dest->numRows(), kSize);
  EXPECT_EQ(table->numDistinct(), kSize);

  // Every key resolves to its relocated row across all runs.
  probe(*table, lookup, keys);
  EXPECT_TRUE(lookup.newGroups.empty());
  for (vector_size_t i = 0; i < kSize; i += 997) {
    EXPECT_EQ(
        keyAt(dest, lookup.hits[i]), keys->asFlatVector<int64_t>()->valueAt(i));
  }
}

} // namespace
} // namespace facebook::velox::exec::test
