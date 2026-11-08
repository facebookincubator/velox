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

#include "velox/exec/JoinTableBuilder.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;

namespace facebook::velox::exec::test {
namespace {

class JoinTableBuilderTest : public testing::Test,
                             public velox::test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  // Options for a build side of 'k BIGINT, v VARCHAR' joined on 'k'.
  JoinTableBuilder::Options makeOptions() const {
    JoinTableBuilder::Options options;
    options.inputType = ROW({"k", "v"}, {BIGINT(), VARCHAR()});
    options.keyChannels = {0};
    return options;
  }

  RowVectorPtr makeInput() {
    return makeRowVector(
        {makeFlatVector<int64_t>({1, 2, 3}),
         makeFlatVector<std::string>({"a", "b", "c"})});
  }

  // Returns the rows of 'table' in insertion order.
  static std::vector<char*> listRows(const BaseHashTable& table) {
    std::vector<char*> rows(table.rows()->numRows());
    RowContainerIterator iter;
    table.rows()->listRows(&iter, rows.size(), rows.data());
    return rows;
  }
};

TEST_F(JoinTableBuilderTest, addInput) {
  const auto options = makeOptions();
  JoinTableBuilder builder(core::JoinType::kInner, options);
  builder.initialize(pool(), pool());

  EXPECT_THAT(builder.tableInputChannels(), testing::ElementsAre(0, 1));
  EXPECT_TRUE(builder.tableType()->equivalent(*options.inputType));

  const auto input = makeInput();
  ASSERT_TRUE(builder.addInput(input));
  ASSERT_TRUE(builder.addInput(input));
  EXPECT_EQ(builder.table()->rows()->numRows(), 2 * input->size());
  EXPECT_FALSE(builder.joinHasNullKeys());
  EXPECT_EQ(builder.numNullKeyRows(), 0);

  // The table the builder hands over is a usable join build side. A join build
  // counts every row, duplicate keys included.
  auto table = builder.takeTable();
  EXPECT_EQ(builder.table(), nullptr);
  table->prepareJoinTable(
      {},
      BaseHashTable::kNoSpillInputStartPartitionBit,
      options.vectorHasherMaxNumDistinct);
  EXPECT_EQ(table->numDistinct(), 2 * input->size());
}

TEST_F(JoinTableBuilderTest, addInputBeforeInitialize) {
  JoinTableBuilder builder(core::JoinType::kInner, makeOptions());
  VELOX_ASSERT_THROW(
      builder.addInput(makeInput()), "JoinTableBuilder is not initialized");
}

TEST_F(JoinTableBuilderTest, dropDuplicates) {
  // A left semi join without a filter only stores the keys, once each.
  JoinTableBuilder builder(core::JoinType::kLeftSemiFilter, [&] {
    auto options = makeOptions();
    // Keep deduplicating the input rather than abandoning it right away.
    options.abandonHashBuildDedupMinPct = 100;
    return options;
  }());
  builder.initialize(pool(), pool());

  EXPECT_THAT(builder.tableInputChannels(), testing::ElementsAre(0));
  EXPECT_TRUE(builder.tableType()->equivalent(*ROW({BIGINT()})));

  const auto input = makeInput();
  ASSERT_TRUE(builder.addInput(input));
  ASSERT_TRUE(builder.addInput(input));
  EXPECT_EQ(builder.table()->rows()->numRows(), input->size());
}

TEST_F(JoinTableBuilderTest, repeatedKeyChannel) {
  // The same column is used by two join keys, e.g. t.k1 = u.k AND t.k2 = u.k.
  // The table stores it once per key.
  auto options = makeOptions();
  options.keyChannels = {0, 0};
  JoinTableBuilder builder(core::JoinType::kInner, std::move(options));
  builder.initialize(pool(), pool());

  EXPECT_THAT(builder.tableInputChannels(), testing::ElementsAre(0, 0, 1));
  EXPECT_TRUE(
      builder.tableType()->equivalent(*ROW({BIGINT(), BIGINT(), VARCHAR()})));

  const auto input = makeInput();
  ASSERT_TRUE(builder.addInput(input));
  EXPECT_EQ(builder.table()->rows()->numRows(), input->size());
}

TEST_F(JoinTableBuilderTest, invalidKeyChannel) {
  auto options = makeOptions();
  options.keyChannels = {2};
  VELOX_ASSERT_THROW(
      JoinTableBuilder(core::JoinType::kInner, options), "(2 vs. 2)");
}

TEST_F(JoinTableBuilderTest, numNullKeyRows) {
  const auto input = makeRowVector(
      {makeNullableFlatVector<int64_t>({1, std::nullopt, 3, std::nullopt}),
       makeFlatVector<std::string>({"a", "b", "c", "d"})});

  // An inner join drops the rows with a null key.
  {
    JoinTableBuilder builder(core::JoinType::kInner, makeOptions());
    builder.initialize(pool(), pool());
    ASSERT_TRUE(builder.addInput(input));
    ASSERT_TRUE(builder.addInput(input));
    EXPECT_EQ(builder.numNullKeyRows(), 4);
    EXPECT_EQ(builder.table()->rows()->numRows(), 4);
    EXPECT_FALSE(builder.joinHasNullKeys());
  }

  // A right join retains them, and they are counted all the same.
  {
    JoinTableBuilder builder(core::JoinType::kRight, makeOptions());
    builder.initialize(pool(), pool());
    ASSERT_TRUE(builder.addInput(input));
    ASSERT_TRUE(builder.addInput(input));
    EXPECT_EQ(builder.numNullKeyRows(), 4);
    EXPECT_EQ(builder.table()->rows()->numRows(), 8);
    EXPECT_FALSE(builder.joinHasNullKeys());
  }
}

TEST_F(JoinTableBuilderTest, nullAwareAntiJoinStopsOnNullKey) {
  auto options = makeOptions();
  options.nullAware = true;
  JoinTableBuilder builder(core::JoinType::kAnti, std::move(options));
  builder.initialize(pool(), pool());

  const auto input = makeRowVector(
      {makeNullableFlatVector<int64_t>({1, std::nullopt}),
       makeFlatVector<std::string>({"a", "b"})});

  // A null build side key makes the join return no rows, so the build can stop.
  EXPECT_FALSE(builder.addInput(input));
  EXPECT_TRUE(builder.joinHasNullKeys());
  EXPECT_EQ(builder.numNullKeyRows(), 1);
}

TEST_F(JoinTableBuilderTest, beforeInsertRows) {
  int32_t numCalls{0};
  auto options = makeOptions();
  options.beforeInsertRows = [&](const RowVectorPtr& input,
                                 SelectivityVector& rows) {
    ++numCalls;
    EXPECT_EQ(rows.size(), input->size());
    // Rows with a null key are dropped before the callback.
    EXPECT_EQ(rows.countSelected(), 2);
    // Deselects the last row, which is then not inserted.
    rows.setValid(input->size() - 1, false);
  };
  JoinTableBuilder builder(core::JoinType::kInner, std::move(options));
  builder.initialize(pool(), pool());

  ASSERT_TRUE(builder.addInput(makeRowVector(
      {makeNullableFlatVector<int64_t>({std::nullopt, 1, 2}),
       makeFlatVector<std::string>({"a", "b", "c"})})));
  EXPECT_EQ(numCalls, 1);

  const auto rows = listRows(*builder.table());
  ASSERT_EQ(rows.size(), 1);
  auto keys = BaseVector::create(BIGINT(), rows.size(), pool());
  builder.table()->rows()->extractColumn(rows.data(), rows.size(), 0, keys);
  velox::test::assertEqualVectors(makeFlatVector<int64_t>({1}), keys);

  // The callback is not invoked if no row is left to insert.
  ASSERT_TRUE(builder.addInput(makeRowVector(
      {makeNullableFlatVector<int64_t>({std::nullopt}),
       makeFlatVector<std::string>({"a"})})));
  EXPECT_EQ(numCalls, 1);
}

TEST_F(JoinTableBuilderTest, probedFlagChannel) {
  auto options = makeOptions();
  options.inputType =
      ROW({"k", "probed", "v"}, {BIGINT(), BOOLEAN(), VARCHAR()});
  options.probedFlagChannel = 1;
  JoinTableBuilder builder(core::JoinType::kRight, std::move(options));
  builder.initialize(pool(), pool());

  // The probed flag column is not a table column.
  EXPECT_THAT(builder.tableInputChannels(), testing::ElementsAre(0, 2));
  EXPECT_TRUE(builder.tableType()->equivalent(*ROW({BIGINT(), VARCHAR()})));

  ASSERT_TRUE(builder.addInput(makeRowVector(
      {makeFlatVector<int64_t>({1, 2, 3}),
       makeFlatVector<bool>({true, false, true}),
       makeFlatVector<std::string>({"a", "b", "c"})})));

  const auto rows = listRows(*builder.table());
  ASSERT_EQ(rows.size(), 3);
  auto probedFlags = BaseVector::create(BOOLEAN(), rows.size(), pool());
  builder.table()->rows()->extractProbedFlags(
      rows.data(), rows.size(), false, false, probedFlags);
  velox::test::assertEqualVectors(
      makeFlatVector<bool>({true, false, true}), probedFlags);
}

TEST_F(JoinTableBuilderTest, invalidProbedFlagChannel) {
  auto options = makeOptions();
  options.probedFlagChannel = 1;
  VELOX_ASSERT_THROW(
      JoinTableBuilder(core::JoinType::kRight, options),
      "The probed flag column must be boolean");

  options.inputType = ROW({"k", "v"}, {BOOLEAN(), VARCHAR()});
  options.probedFlagChannel = 0;
  VELOX_ASSERT_THROW(
      JoinTableBuilder(core::JoinType::kRight, options),
      "The probed flag column can not be a join key");
}

} // namespace
} // namespace facebook::velox::exec::test
