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

#include "velox/exec/VectorHasher.h"
#include "velox/experimental/cxl/CxlHashTable.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;

namespace facebook::velox::cxl {
namespace {

// Drives a CxlHashTable directly through groupProbe to exercise
// relocateAllToCxl without the operator or a real CXL device: the "CXL"
// container lives on an ordinary second memory pool, enough for the
// bucket-swizzle seam under test.
class CxlHashTableTest : public testing::Test,
                         public velox::test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  void SetUp() override {
    cxlPool_ = memory::memoryManager()->addLeafPool("cxl-table-test");
  }

  // Builds a single-bigint-key aggregation table with no accumulators and a
  // second container on 'cxlPool_'.
  std::unique_ptr<CxlHashTable<false>> makeTable() {
    std::vector<std::unique_ptr<exec::VectorHasher>> hashers;
    hashers.push_back(exec::VectorHasher::create(BIGINT(), 0));
    return CxlHashTable<false>::createForAggregation(
        std::move(hashers), {}, pool_.get(), cxlPool_.get());
  }

  // Probes 'keys'; hits and newGroups land in 'lookup'.
  void probe(
      CxlHashTable<false>& table,
      exec::HashLookup& lookup,
      const VectorPtr& keys) {
    auto input = makeRowVector({keys});
    SelectivityVector rows(keys->size());
    table.prepareForGroupProbe(
        lookup,
        input,
        rows,
        exec::BaseHashTable::kNoSpillInputStartPartitionBit);
    table.groupProbe(
        lookup, exec::BaseHashTable::kNoSpillInputStartPartitionBit);
  }

  // More distinct keys than VectorHasher::kMaxDistinct, spread wide, so the
  // hasher cannot assign value ids and the table uses a bucketed hash mode.
  static constexpr vector_size_t kBucketedSize = 150'000;

  VectorPtr wideKeys(vector_size_t size) {
    return makeFlatVector<int64_t>(
        size, [](auto row) { return row * 1'000'003LL + 7; });
  }

  // Extracts the key column of 'row' from 'container'.
  int64_t keyAt(exec::RowContainer* container, char* row) {
    auto result = BaseVector::create(BIGINT(), 1, pool_.get());
    container->extractColumn(&row, 1, 0, result);
    return result->asFlatVector<int64_t>()->valueAt(0);
  }

  std::shared_ptr<memory::MemoryPool> cxlPool_;
};

TEST_F(CxlHashTableTest, relocateAllToCxlAndReprobe) {
  constexpr vector_size_t kSize = kBucketedSize;
  auto table = makeTable();
  exec::HashLookup lookup(table->hashers(), pool_.get());
  auto keys = wideKeys(kSize);

  probe(*table, lookup, keys);
  ASSERT_EQ(lookup.newGroups.size(), kSize);
  ASSERT_NE(table->hashMode(), exec::BaseHashTable::HashMode::kArray);

  table->relocateAllToCxl();
  EXPECT_EQ(table->rows()->numRows(), 0);
  EXPECT_EQ(table->cxlRows()->numRows(), kSize);
  EXPECT_EQ(table->numDistinct(), kSize);

  // Every key resolves to its relocated CXL row, creating no new groups.
  probe(*table, lookup, keys);
  EXPECT_TRUE(lookup.newGroups.empty());
  for (vector_size_t i = 0; i < kSize; i += 1'000) {
    EXPECT_EQ(
        keyAt(table->cxlRows(), lookup.hits[i]),
        keys->asFlatVector<int64_t>()->valueAt(i));
  }
}

TEST_F(CxlHashTableTest, relocateInArrayMode) {
  constexpr vector_size_t kSize = 1'000;
  auto table = makeTable();
  exec::HashLookup lookup(table->hashers(), pool_.get());
  // A dense small range keeps the table in kArray mode.
  auto keys = makeFlatVector<int64_t>(kSize, [](auto row) { return row; });

  probe(*table, lookup, keys);
  ASSERT_EQ(table->hashMode(), exec::BaseHashTable::HashMode::kArray);
  table->relocateAllToCxl();
  EXPECT_EQ(table->cxlRows()->numRows(), kSize);

  probe(*table, lookup, keys);
  EXPECT_TRUE(lookup.newGroups.empty());
  constexpr vector_size_t kRow = 42;
  EXPECT_EQ(
      keyAt(table->cxlRows(), lookup.hits[kRow]),
      keys->asFlatVector<int64_t>()->valueAt(kRow));
}

TEST_F(CxlHashTableTest, relocateSurvivesRehash) {
  constexpr vector_size_t kSize = 1'000;
  auto table = makeTable();
  exec::HashLookup lookup(table->hashers(), pool_.get());
  auto keys = wideKeys(kSize);

  probe(*table, lookup, keys);
  table->relocateAllToCxl();

  // Grow well past capacity to force a rehash. The rebuild lists rows from both
  // containers and reindexes the relocated rows from the CXL container.
  constexpr vector_size_t kGrow = 100'000;
  auto growKeys = makeFlatVector<int64_t>(
      kGrow, [](auto row) { return -(row * 999'983LL + 11); });
  probe(*table, lookup, growKeys);
  EXPECT_EQ(lookup.newGroups.size(), kGrow);

  probe(*table, lookup, keys);
  EXPECT_TRUE(lookup.newGroups.empty());
  EXPECT_EQ(table->numDistinct(), kSize + kGrow);
}

} // namespace
} // namespace facebook::velox::cxl
