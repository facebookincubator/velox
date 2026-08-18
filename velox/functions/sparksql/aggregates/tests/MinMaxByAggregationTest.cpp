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

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/Aggregate.h"
#include "velox/exec/RowContainer.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/lib/aggregates/tests/utils/AggregationTestBase.h"
#include "velox/functions/sparksql/aggregates/Register.h"

using namespace facebook::velox::functions::aggregate::test;

namespace facebook::velox::functions::aggregate::sparksql::test {

namespace {

class MinMaxByAggregateTest : public AggregationTestBase {
 protected:
  void SetUp() override {
    AggregationTestBase::SetUp();
    registerAggregateFunctions("spark_");
  }
};

TEST_F(MinMaxByAggregateTest, maxBy) {
  auto vectors = {makeRowVector({
      makeFlatVector<int32_t>({1, 2, 3}),
      makeFlatVector<int32_t>({11, 12, 12}),
  })};

  auto expected = {makeRowVector({
      makeFlatVector<int32_t>(std::vector<int32_t>({3})),
  })};

  testAggregations(vectors, {}, {"spark_max_by(c0, c1)"}, expected);
}

TEST_F(MinMaxByAggregateTest, minBy) {
  auto vectors = {makeRowVector({
      makeFlatVector<int32_t>({1, 2, 3}),
      makeFlatVector<int32_t>({12, 11, 11}),
  })};

  auto expected = {makeRowVector({
      makeFlatVector<int32_t>(std::vector<int32_t>({3})),
  })};

  testAggregations(vectors, {}, {"spark_min_by(c0, c1)"}, expected);
}

// Drives the aggregate directly so the per-row variable-length counter that
// RowContainer keeps at 'rowSizeOffset' can be read back. Returns that counter
// after aggregating 'values'/'comparisons' into a single group.
uint32_t aggregateAndReadRowSize(
    memory::MemoryPool* pool,
    const std::string& name,
    const TypePtr& valueType,
    const VectorPtr& values,
    const VectorPtr& comparisons) {
  core::QueryConfig queryConfig({});
  auto fn = exec::Aggregate::create(
      name,
      core::AggregationNode::Step::kSingle,
      std::vector<TypePtr>{valueType, INTEGER()},
      valueType,
      queryConfig);

  HashStringAllocator stringAllocator{pool};
  fn->setAllocator(&stringAllocator);

  const int32_t rowSizeOffset = bits::nbytes(1);
  int32_t offset = rowSizeOffset + sizeof(uint32_t);
  offset = bits::roundUp(offset, fn->accumulatorAlignmentSize());
  fn->setOffsets(
      offset,
      exec::RowContainer::nullByte(0),
      exec::RowContainer::nullMask(0),
      exec::RowContainer::initializedByte(0),
      exec::RowContainer::initializedMask(0),
      rowSizeOffset);

  const auto size = values->size();
  std::vector<char> group(offset + fn->accumulatorFixedWidthSize());
  std::vector<char*> groups(size, group.data());
  std::vector<vector_size_t> indices{0};
  fn->initializeNewGroups(groups.data(), indices);

  SelectivityVector rows{size};
  fn->addRawInput(groups.data(), rows, {values, comparisons}, false);

  return *reinterpret_cast<uint32_t*>(group.data() + rowSizeOffset);
}

TEST_F(MinMaxByAggregateTest, trackRowSize) {
  // Spark registers min_by/max_by on the same MinMaxByAggregateBase as Presto
  // but with its own comparator, so it instantiates different specializations.
  // A non-numeric value lives in the HashStringAllocator behind a
  // SingleValueAccumulator, and the row's variable-length size must grow with
  // it or Spiller::extractSpillVector sizes spill batches from a number that
  // is orders of magnitude too small.
  auto comparisons = makeFlatVector<int32_t>({5, 4, 3, 2, 1});

  auto arrays = makeArrayVector<int64_t>({
      {1, 2, 3},
      {4, 5},
      {6},
      {7, 8, 9, 10},
      {11},
  });
  // Numeric value and comparison: the accumulator is genuinely fixed size, so
  // nothing is tracked. This is also what shows the counter is only ever moved
  // by the tracker.
  auto numbers = makeFlatVector<int32_t>({1, 2, 3, 4, 5});

  for (const auto& name : {"spark_min_by", "spark_max_by"}) {
    SCOPED_TRACE(name);
    EXPECT_GT(
        aggregateAndReadRowSize(
            pool(), name, ARRAY(BIGINT()), arrays, comparisons),
        0);
    EXPECT_EQ(
        aggregateAndReadRowSize(pool(), name, INTEGER(), numbers, comparisons),
        0);
  }
}

TEST_F(MinMaxByAggregateTest, arrayCompare) {
  auto data = makeRowVector({
      makeArrayVector<int64_t>({
          {1, 2, 3},
          {4, 5},
          {6, 7, 8},
      }),
      makeNullableArrayVector<int64_t>({
          {4, 5},
          {std::nullopt, 2},
          {6, 7, 8},
      }),
  });

  auto expected = makeRowVector({
      makeArrayVector<int64_t>({
          {4, 5},
      }),
      makeArrayVector<int64_t>({
          {6, 7, 8},
      }),
  });

  testAggregations(
      {data}, {}, {"spark_min_by(c0, c1)", "spark_max_by(c0, c1)"}, {expected});

  data = makeRowVector({
      makeArrayVector<int64_t>({
          {1, 2, 3},
          {4, 5},
          {6, 7, 8},
      }),
      makeNullableArrayVector<int64_t>({
          {1, 2, 3},
          {3, std::nullopt, 4},
          {6, 7, 8},
      }),
  });

  expected = makeRowVector({
      makeArrayVector<int64_t>({
          {1, 2, 3},
      }),
      makeArrayVector<int64_t>({
          {6, 7, 8},
      }),
  });

  testAggregations(
      {data}, {}, {"spark_min_by(c0, c1)", "spark_max_by(c0, c1)"}, {expected});
}

TEST_F(MinMaxByAggregateTest, mapCompare) {
  auto data = makeRowVector({
      makeArrayVector<int64_t>({}),
      makeNullableMapVector<int64_t, int64_t>({}),
  });
  std::vector<RowVectorPtr> expected = {};

  VELOX_ASSERT_USER_THROW(
      testAggregations({data}, {}, {"spark_min_by(c0, c1)"}, expected),
      "Aggregate function signature is not supported");

  VELOX_ASSERT_USER_THROW(
      testAggregations({data}, {}, {"spark_max_by(c0, c1)"}, expected),
      "Aggregate function signature is not supported");
}

TEST_F(MinMaxByAggregateTest, rowCompare) {
  auto data = makeRowVector({
      makeArrayVector<int64_t>({
          {1, 2, 3},
          {4, 5},
          {6, 7, 8},
      }),
      makeRowVector({makeNullableFlatVector<int32_t>({
          1,
          std::nullopt,
          3,
      })}),
  });

  auto expected = makeRowVector({
      makeArrayVector<int64_t>({
          {4, 5},
      }),
      makeArrayVector<int64_t>({
          {6, 7, 8},
      }),
  });

  testAggregations(
      {data}, {}, {"spark_min_by(c0, c1)", "spark_max_by(c0, c1)"}, {expected});

  data = makeRowVector({
      makeArrayVector<int64_t>({
          {1, 2, 3},
          {4, 5},
          {6, 7, 8},
      }),
      makeRowVector({makeNullableFlatVector<int32_t>({
          1,
          2,
          3,
      })}),
  });

  expected = makeRowVector({
      makeArrayVector<int64_t>({
          {1, 2, 3},
      }),
      makeArrayVector<int64_t>({
          {6, 7, 8},
      }),
  });

  testAggregations(
      {data}, {}, {"spark_min_by(c0, c1)", "spark_max_by(c0, c1)"}, {expected});
}

} // namespace
} // namespace facebook::velox::functions::aggregate::sparksql::test
