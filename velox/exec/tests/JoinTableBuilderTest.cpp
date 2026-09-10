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
  JoinTableBuilder::Options makeOptions(core::JoinType joinType) const {
    JoinTableBuilder::Options options;
    options.joinType = joinType;
    options.inputType = ROW({"k", "v"}, {BIGINT(), VARCHAR()});
    options.joinKeys = {
        std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "k"),
    };
    return options;
  }

  RowVectorPtr makeInput() {
    return makeRowVector(
        {makeFlatVector<int64_t>({1, 2, 3}),
         makeFlatVector<std::string>({"a", "b", "c"})});
  }
};

TEST_F(JoinTableBuilderTest, addInput) {
  JoinTableBuilder builder(makeOptions(core::JoinType::kInner));
  builder.initialize(pool(), pool());

  EXPECT_THAT(builder.keyChannels(), testing::ElementsAre(0));
  EXPECT_THAT(builder.dependentChannels(), testing::ElementsAre(1));
  EXPECT_FALSE(builder.dropDuplicates());

  const auto input = makeInput();
  ASSERT_TRUE(builder.addInput(input));
  ASSERT_TRUE(builder.addInput(input));
  EXPECT_EQ(builder.table()->rows()->numRows(), 2 * input->size());
  EXPECT_FALSE(builder.joinHasNullKeys());

  // The table the builder hands over is a usable join build side. A join build
  // counts every row, duplicate keys included.
  auto table = builder.takeTable();
  table->prepareJoinTable(
      {},
      BaseHashTable::kNoSpillInputStartPartitionBit,
      builder.vectorHasherMaxNumDistinct());
  EXPECT_EQ(table->numDistinct(), 2 * input->size());
}

TEST_F(JoinTableBuilderTest, phasesMustRunInOrder) {
  JoinTableBuilder builder(makeOptions(core::JoinType::kInner));
  builder.initialize(pool(), pool());
  const auto input = makeInput();

  // No input has been decoded yet, so every phase but the first is rejected.
  VELOX_ASSERT_THROW(builder.processNullKeys(), "must be called in order");
  VELOX_ASSERT_THROW(
      builder.decodeDependents(input), "must be called in order");
  VELOX_ASSERT_THROW(builder.insertRows(input), "must be called in order");

  builder.decodeKeys(input);
  VELOX_ASSERT_THROW(builder.insertRows(input), "must be called in order");

  // A rejected phase leaves the sequence where it was, and decodeKeys() starts
  // it over, so the input can still be processed.
  builder.decodeKeys(input);
  ASSERT_TRUE(builder.processNullKeys());
  VELOX_ASSERT_THROW(builder.insertRows(input), "must be called in order");
  builder.decodeDependents(input);
  builder.insertRows(input);
  EXPECT_EQ(builder.table()->rows()->numRows(), input->size());

  // The input has been consumed, so inserting it again is rejected rather than
  // silently duplicating its rows.
  VELOX_ASSERT_THROW(builder.insertRows(input), "must be called in order");
  EXPECT_EQ(builder.table()->rows()->numRows(), input->size());
}

TEST_F(JoinTableBuilderTest, nullAwareAntiJoinStopsOnNullKey) {
  auto options = makeOptions(core::JoinType::kAnti);
  options.nullAware = true;
  JoinTableBuilder builder(std::move(options));
  builder.initialize(pool(), pool());

  const auto input = makeRowVector(
      {makeNullableFlatVector<int64_t>({1, std::nullopt}),
       makeFlatVector<std::string>({"a", "b"})});

  // A null build side key makes the join return no rows, so the build can stop.
  EXPECT_FALSE(builder.addInput(input));
  EXPECT_TRUE(builder.joinHasNullKeys());
}

} // namespace
} // namespace facebook::velox::exec::test
