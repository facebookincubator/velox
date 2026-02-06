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
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/functions/lib/aggregates/tests/utils/AggregationTestBase.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::functions::aggregate::test;

namespace facebook::velox::aggregate::test {

namespace {

class ArrayUnionSumTest : public AggregationTestBase {};

TEST_F(ArrayUnionSumTest, global) {
  const std::vector<std::vector<std::optional<int64_t>>> inputData = {
      {{1, 2, 3}},
      {{10, 5, 4, 1}},
      {{9, std::nullopt, 5, 4}},
  };
  auto data = makeRowVector({makeNullableArrayVector(inputData)});

  auto expected = makeRowVector({
      makeArrayVector<int64_t>({
          {20, 7, 12, 5},
      }),
  });

  testAggregations({data}, {}, {"array_union_sum(c0)"}, {expected});
}

TEST_F(ArrayUnionSumTest, globalWithNulls) {
  const std::vector<std::vector<std::optional<int64_t>>> inputData = {
      {{1, std::nullopt, 3}},
      {{10, 5, std::nullopt}},
      {{std::nullopt, std::nullopt, 5}},
  };
  auto data = makeRowVector({makeNullableArrayVector(inputData)});

  auto expected = makeRowVector({
      makeArrayVector<int64_t>({
          {11, 5, 8},
      }),
  });

  testAggregations({data}, {}, {"array_union_sum(c0)"}, {expected});
}

TEST_F(ArrayUnionSumTest, globalDifferentLengths) {
  const std::vector<std::vector<std::optional<int64_t>>> inputData = {
      {{1, 2}},
      {{10, 5, 4, 1}},
      {{9}},
  };
  auto data = makeRowVector({makeNullableArrayVector(inputData)});

  auto expected = makeRowVector({
      makeArrayVector<int64_t>({
          {20, 7, 4, 1},
      }),
  });

  testAggregations({data}, {}, {"array_union_sum(c0)"}, {expected});
}

TEST_F(ArrayUnionSumTest, emptyArrays) {
  auto allEmptyArrays = makeRowVector({
      makeArrayVector<int64_t>({
          {},
          {},
          {},
      }),
  });

  auto expectedEmpty = makeRowVector({
      makeArrayVector<int64_t>({
          {},
      }),
  });

  testAggregations(
      {allEmptyArrays}, {}, {"array_union_sum(c0)"}, {expectedEmpty});
}

TEST_F(ArrayUnionSumTest, nullArrays) {
  auto allNullArrays = makeRowVector(
      {BaseVector::createNullConstant(ARRAY(BIGINT()), 3, pool())});

  auto expectedNull = makeRowVector(
      {BaseVector::createNullConstant(ARRAY(BIGINT()), 1, pool())});

  testAggregations(
      {allNullArrays}, {}, {"array_union_sum(c0)"}, {expectedNull});
}

TEST_F(ArrayUnionSumTest, tinyintOverflow) {
  const std::vector<std::vector<std::optional<int8_t>>> inputData = {
      {{10, 20}},
      {{100, 30}},
      {{30, 50}},
  };
  auto data = makeRowVector({makeNullableArrayVector(inputData)});

  auto plan = PlanBuilder()
                  .values({data})
                  .singleAggregation({}, {"array_union_sum(c0)"})
                  .planNode();

  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).copyResults(pool()), "Value 140 exceeds 127");

  const std::vector<std::vector<std::optional<int8_t>>> negInputData = {
      {{-10, -20}},
      {{-100, -30}},
      {{-30, -50}},
  };
  data = makeRowVector({makeNullableArrayVector(negInputData)});

  plan = PlanBuilder()
             .values({data})
             .singleAggregation({}, {"array_union_sum(c0)"})
             .planNode();

  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).copyResults(pool()),
      "Value -140 is less than -128");
}

TEST_F(ArrayUnionSumTest, smallintOverflow) {
  const int16_t largeValue = std::numeric_limits<int16_t>::max() - 20;
  const int16_t smallValue = std::numeric_limits<int16_t>::min() + 20;

  const std::vector<std::vector<std::optional<int16_t>>> inputData = {
      {{10, 20}},
      {{largeValue, 30}},
      {{30, 50}},
  };
  auto data = makeRowVector({makeNullableArrayVector(inputData)});

  auto plan = PlanBuilder()
                  .values({data})
                  .singleAggregation({}, {"array_union_sum(c0)"})
                  .planNode();

  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).copyResults(pool()),
      "Value 32787 exceeds 32767");

  const std::vector<std::vector<std::optional<int16_t>>> negInputData = {
      {{-10, -20}},
      {{smallValue, -30}},
      {{-30, -50}},
  };
  data = makeRowVector({makeNullableArrayVector(negInputData)});

  plan = PlanBuilder()
             .values({data})
             .singleAggregation({}, {"array_union_sum(c0)"})
             .planNode();

  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).copyResults(pool()),
      "Value -32788 is less than -32768");
}

TEST_F(ArrayUnionSumTest, integerOverflow) {
  const int32_t largeValue = std::numeric_limits<int32_t>::max() - 20;
  const int32_t smallValue = std::numeric_limits<int32_t>::min() + 20;

  const std::vector<std::vector<std::optional<int32_t>>> inputData = {
      {{10, 20}},
      {{largeValue, 30}},
      {{30, 50}},
  };
  auto data = makeRowVector({makeNullableArrayVector(inputData)});

  auto plan = PlanBuilder()
                  .values({data})
                  .singleAggregation({}, {"array_union_sum(c0)"})
                  .planNode();

  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).copyResults(pool()),
      "Value 2147483667 exceeds 2147483647");

  const std::vector<std::vector<std::optional<int32_t>>> negInputData = {
      {{-10, -20}},
      {{smallValue, -30}},
      {{-30, -50}},
  };
  data = makeRowVector({makeNullableArrayVector(negInputData)});

  plan = PlanBuilder()
             .values({data})
             .singleAggregation({}, {"array_union_sum(c0)"})
             .planNode();

  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).copyResults(pool()),
      "Value -2147483668 is less than -2147483648");
}

TEST_F(ArrayUnionSumTest, bigintOverflow) {
  const int64_t largeValue = std::numeric_limits<int64_t>::max() - 20;
  const int64_t smallValue = std::numeric_limits<int64_t>::min() + 20;

  const std::vector<std::vector<std::optional<int64_t>>> inputData = {
      {{10, 20}},
      {{largeValue, 30}},
      {{30, 50}},
  };
  auto data = makeRowVector({makeNullableArrayVector(inputData)});

  auto plan = PlanBuilder()
                  .values({data})
                  .singleAggregation({}, {"array_union_sum(c0)"})
                  .planNode();

  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).copyResults(pool()),
      "Value 9223372036854775827 exceeds 9223372036854775807");

  const std::vector<std::vector<std::optional<int64_t>>> negInputData = {
      {{-10, -20}},
      {{smallValue, -30}},
      {{-30, -50}},
  };
  data = makeRowVector({makeNullableArrayVector(negInputData)});

  plan = PlanBuilder()
             .values({data})
             .singleAggregation({}, {"array_union_sum(c0)"})
             .planNode();

  VELOX_ASSERT_THROW(
      AssertQueryBuilder(plan).copyResults(pool()),
      "Value -9223372036854775828 is less than -9223372036854775808");
}

TEST_F(ArrayUnionSumTest, floatNan) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  constexpr float kNan = std::numeric_limits<float>::quiet_NaN();

  const std::vector<std::vector<std::optional<float>>> inputData = {
      {{10.0F, 20.0F}},
      {{kNan, 30.0F}},
      {{30.0F, kInf}},
  };
  auto data = makeRowVector({makeNullableArrayVector(inputData)});

  auto expected = makeRowVector({
      makeArrayVector<float>({
          {kNan, kInf},
      }),
  });

  testAggregations({data}, {}, {"array_union_sum(c0)"}, {expected});
}

TEST_F(ArrayUnionSumTest, doubleNan) {
  constexpr double kInf = std::numeric_limits<double>::infinity();
  constexpr double kNan = std::numeric_limits<double>::quiet_NaN();

  const std::vector<std::vector<std::optional<double>>> inputData = {
      {{10.0, 20.0}},
      {{kNan, 30.0}},
      {{30.0, kInf}},
  };
  auto data = makeRowVector({makeNullableArrayVector(inputData)});

  auto expected = makeRowVector({
      makeArrayVector<double>({
          {kNan, kInf},
      }),
  });

  testAggregations({data}, {}, {"array_union_sum(c0)"}, {expected});
}

TEST_F(ArrayUnionSumTest, groupBy) {
  const std::vector<std::vector<std::optional<int64_t>>> inputData1 = {
      {{1, 2, 3}},
  };
  const std::vector<std::vector<std::optional<int64_t>>> inputData2 = {
      {{10, 5, 4}},
  };
  const std::vector<std::vector<std::optional<int64_t>>> inputData3 = {
      {{1, 2}},
  };
  const std::vector<std::vector<std::optional<int64_t>>> inputData4 = {
      {{9, 0, 5}},
  };

  // Create multiple batches for group by
  auto batch1 = makeRowVector({
      makeFlatVector<int64_t>({1}),
      makeNullableArrayVector(inputData1),
  });
  auto batch2 = makeRowVector({
      makeFlatVector<int64_t>({1}),
      makeNullableArrayVector(inputData2),
  });
  auto batch3 = makeRowVector({
      makeFlatVector<int64_t>({2}),
      makeNullableArrayVector(inputData3),
  });
  auto batch4 = makeRowVector({
      makeFlatVector<int64_t>({1}),
      makeNullableArrayVector(inputData4),
  });

  auto expected = makeRowVector({
      makeFlatVector<int64_t>({1, 2}),
      makeArrayVector<int64_t>({
          {20, 7, 12},
          {1, 2},
      }),
  });

  testAggregations(
      {batch1, batch2, batch3, batch4},
      {"c0"},
      {"array_union_sum(c1)"},
      {expected});
}

TEST_F(ArrayUnionSumTest, singleElement) {
  const std::vector<std::vector<std::optional<int64_t>>> inputData = {
      {{100}},
      {{200}},
      {{300}},
  };
  auto data = makeRowVector({makeNullableArrayVector(inputData)});

  auto expected = makeRowVector({
      makeArrayVector<int64_t>({
          {600},
      }),
  });

  testAggregations({data}, {}, {"array_union_sum(c0)"}, {expected});
}

TEST_F(ArrayUnionSumTest, allNullElements) {
  const std::vector<std::vector<std::optional<int64_t>>> inputData = {
      {{std::nullopt, std::nullopt}},
      {{std::nullopt, std::nullopt}},
      {{std::nullopt, std::nullopt}},
  };
  auto data = makeRowVector({makeNullableArrayVector(inputData)});

  auto expected = makeRowVector({
      makeArrayVector<int64_t>({
          {0, 0},
      }),
  });

  testAggregations({data}, {}, {"array_union_sum(c0)"}, {expected});
}

TEST_F(ArrayUnionSumTest, realType) {
  const std::vector<std::vector<std::optional<float>>> inputData = {
      {{1.5F, 2.5F}},
      {{10.5F, 5.5F, 4.0F}},
      {{9.0F, std::nullopt, 5.5F}},
  };
  auto data = makeRowVector({makeNullableArrayVector(inputData)});

  auto expected = makeRowVector({
      makeArrayVector<float>({
          {21.0F, 8.0F, 9.5F},
      }),
  });

  testAggregations({data}, {}, {"array_union_sum(c0)"}, {expected});
}

TEST_F(ArrayUnionSumTest, doubleType) {
  const std::vector<std::vector<std::optional<double>>> inputData = {
      {{1.5, 2.5}},
      {{10.5, 5.5, 4.0}},
      {{9.0, std::nullopt, 5.5}},
  };
  auto data = makeRowVector({makeNullableArrayVector(inputData)});

  auto expected = makeRowVector({
      makeArrayVector<double>({
          {21.0, 8.0, 9.5},
      }),
  });

  testAggregations({data}, {}, {"array_union_sum(c0)"}, {expected});
}

} // namespace

// ============================================================================
// CUSTOM FUZZER TESTS
// These tests use VectorFuzzer to generate random inputs and verify
// properties of the array_union_sum aggregation function.
// ============================================================================

class ArrayUnionSumFuzzerTest
    : public functions::aggregate::test::AggregationTestBase {
 protected:
  template <typename T>
  void runFuzzerTest(
      const TypePtr& elementType,
      int vectorSize,
      double nullRatio,
      int iterations) {
    VectorFuzzer::Options opts;
    opts.vectorSize = vectorSize;
    opts.nullRatio = nullRatio;
    opts.containerLength = 10;
    opts.containerVariableLength = true;
    opts.containerHasNulls = true;
    VectorFuzzer fuzzer(opts, pool());

    for (int iter = 0; iter < iterations; ++iter) {
      auto inputArrays = fuzzer.fuzz(ARRAY(elementType));
      auto data = makeRowVector({inputArrays});

      auto plan = exec::test::PlanBuilder()
                      .values({data})
                      .singleAggregation({}, {"array_union_sum(c0)"})
                      .planNode();

      VectorPtr result;
      try {
        result = exec::test::AssertQueryBuilder(plan).copyResults(pool());
      } catch (const VeloxUserError&) {
        // Overflow errors are expected for integer types with random data
        continue;
      }

      if (!result) {
        continue;
      }

      auto resultRow = result->as<RowVector>();
      if (!resultRow || resultRow->size() == 0) {
        continue;
      }

      auto resultArray = resultRow->childAt(0);
      if (resultArray->isNullAt(0)) {
        // Result is null when all input arrays are null
        bool allInputsNull = true;
        for (vector_size_t i = 0; i < inputArrays->size(); ++i) {
          if (!inputArrays->isNullAt(i)) {
            allInputsNull = false;
            break;
          }
        }
        ASSERT_TRUE(allInputsNull)
            << "Result should only be null when all inputs are null";
        continue;
      }

      auto resultArrayVec = resultArray->as<ArrayVector>();
      ASSERT_NE(resultArrayVec, nullptr);

      // Verify: result array length should be max of all input array lengths.
      // We need to decode the input arrays since VectorFuzzer may generate
      // wrapped vectors (Dictionary, Constant, etc.)
      SelectivityVector allRows(inputArrays->size());
      DecodedVector decodedInputArrays(*inputArrays, allRows);
      auto baseArrayVector =
          decodedInputArrays.base()->template as<ArrayVector>();

      vector_size_t maxInputLength = 0;
      if (baseArrayVector) {
        for (vector_size_t i = 0; i < inputArrays->size(); ++i) {
          if (!decodedInputArrays.isNullAt(i)) {
            auto decodedIndex = decodedInputArrays.index(i);
            maxInputLength =
                std::max(maxInputLength, baseArrayVector->sizeAt(decodedIndex));
          }
        }

        ASSERT_EQ(resultArrayVec->sizeAt(0), maxInputLength)
            << "Result array length should equal max input array length";
      }

      // Verify: result array elements are not null (nulls are treated as 0)
      auto resultElements = resultArrayVec->elements();
      auto resultOffset = resultArrayVec->offsetAt(0);
      auto resultSize = resultArrayVec->sizeAt(0);
      for (vector_size_t i = 0; i < resultSize; ++i) {
        ASSERT_FALSE(resultElements->isNullAt(resultOffset + i))
            << "Result elements should never be null";
      }
    }
  }
};

TEST_F(ArrayUnionSumFuzzerTest, fuzzBigint) {
  runFuzzerTest<int64_t>(BIGINT(), 100, 0.1, 50);
}

TEST_F(ArrayUnionSumFuzzerTest, fuzzInteger) {
  runFuzzerTest<int32_t>(INTEGER(), 100, 0.1, 50);
}

TEST_F(ArrayUnionSumFuzzerTest, fuzzSmallint) {
  runFuzzerTest<int16_t>(SMALLINT(), 100, 0.1, 50);
}

TEST_F(ArrayUnionSumFuzzerTest, fuzzTinyint) {
  runFuzzerTest<int8_t>(TINYINT(), 100, 0.1, 50);
}

TEST_F(ArrayUnionSumFuzzerTest, fuzzDouble) {
  runFuzzerTest<double>(DOUBLE(), 100, 0.1, 50);
}

TEST_F(ArrayUnionSumFuzzerTest, fuzzReal) {
  runFuzzerTest<float>(REAL(), 100, 0.1, 50);
}

TEST_F(ArrayUnionSumFuzzerTest, fuzzHighNullRatio) {
  runFuzzerTest<int64_t>(BIGINT(), 100, 0.5, 50);
}

TEST_F(ArrayUnionSumFuzzerTest, fuzzLargeVectors) {
  runFuzzerTest<int64_t>(BIGINT(), 500, 0.1, 20);
}

TEST_F(ArrayUnionSumFuzzerTest, fuzzGroupBy) {
  VectorFuzzer::Options opts;
  opts.vectorSize = 100;
  opts.nullRatio = 0.1;
  opts.containerLength = 10;
  opts.containerVariableLength = true;
  opts.containerHasNulls = true;
  VectorFuzzer fuzzer(opts, pool());

  for (int iter = 0; iter < 20; ++iter) {
    auto groupKeys = fuzzer.fuzz(INTEGER());
    auto inputArrays = fuzzer.fuzz(ARRAY(BIGINT()));
    auto data = makeRowVector({groupKeys, inputArrays});

    auto plan = exec::test::PlanBuilder()
                    .values({data})
                    .singleAggregation({"c0"}, {"array_union_sum(c1)"})
                    .planNode();

    VectorPtr result;
    try {
      result = exec::test::AssertQueryBuilder(plan).copyResults(pool());
    } catch (const VeloxUserError&) {
      // Overflow errors are expected
      continue;
    }

    // Verify result is not empty and has expected structure
    ASSERT_NE(result, nullptr);
    auto resultRow = result->as<RowVector>();
    ASSERT_NE(resultRow, nullptr);
    ASSERT_EQ(resultRow->childrenSize(), 2);
  }
}

TEST_F(ArrayUnionSumFuzzerTest, fuzzEmptyArrays) {
  VectorFuzzer::Options opts;
  opts.vectorSize = 50;
  opts.nullRatio = 0.1;
  opts.containerLength = 0; // Empty arrays
  opts.containerVariableLength = true;
  VectorFuzzer fuzzer(opts, pool());

  for (int iter = 0; iter < 20; ++iter) {
    auto inputArrays = fuzzer.fuzz(ARRAY(BIGINT()));
    auto data = makeRowVector({inputArrays});

    auto plan = exec::test::PlanBuilder()
                    .values({data})
                    .singleAggregation({}, {"array_union_sum(c0)"})
                    .planNode();

    VectorPtr result;
    try {
      result = exec::test::AssertQueryBuilder(plan).copyResults(pool());
    } catch (const VeloxUserError&) {
      continue;
    }

    // Verify result doesn't crash and has valid structure
    ASSERT_NE(result, nullptr);
    auto resultRow = result->as<RowVector>();
    ASSERT_NE(resultRow, nullptr);
  }
}

} // namespace facebook::velox::aggregate::test
