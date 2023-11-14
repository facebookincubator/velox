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
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

using namespace facebook::velox;
using namespace facebook::velox::exec::test;

class UnnestTest : public OperatorTestBase {};

TEST_F(UnnestTest, basicArray) {
  auto vector = makeRowVector({
      makeFlatVector<int64_t>(100, [](auto row) { return row; }),
      makeArrayVector<int32_t>(
          100,
          [](auto row) { return row % 5 + 1; },
          [](auto row, auto index) { return index * (row % 3); },
          nullEvery(7)),
  });

  createDuckDbTable({vector});

  // TODO Add tests with empty arrays. This requires better support in DuckDB.

  auto op = PlanBuilder().values({vector}).unnest({"c0"}, {"c1"}).planNode();
  assertQuery(op, "SELECT c0, UNNEST(c1) FROM tmp WHERE c0 % 7 > 0");

  // Tests with an array with dictionary encoded elements.
  VectorFuzzer::Options opts;
  opts.containerVariableLength = false;
  VectorFuzzer fuzzer(opts, pool());
  vector = makeRowVector(
      {fuzzer.fuzzFlat(INTEGER(), 30),
       fuzzer.fuzzArray(
           fuzzer.fuzzDictionary(fuzzer.fuzzFlat(INTEGER(), 100), 100), 30)});

  createDuckDbTable({vector});
  op = PlanBuilder().values({vector}).unnest({"c0"}, {"c1"}).planNode();
  assertQuery(op, "SELECT c0, UNNEST(c1) FROM tmp");
}

TEST_F(UnnestTest, arrayWithOrdinality) {
  auto array = vectorMaker_.arrayVectorNullable<int32_t>(
      {{{1, 2, std::nullopt, 4}},
       std::nullopt,
       {{5, 6}},
       {{}},
       {{{{std::nullopt}}}},
       {{7, 8, 9}}});
  auto vector = makeRowVector(
      {makeNullableFlatVector<double>({1.1, 2.2, 3.3, 4.4, 5.5, std::nullopt}),
       array});

  auto op = PlanBuilder()
                .values({vector})
                .unnest({"c0"}, {"c1"}, "ordinal")
                .planNode();

  auto expected = makeRowVector(
      {makeNullableFlatVector<double>(
           {1.1,
            1.1,
            1.1,
            1.1,
            3.3,
            3.3,
            5.5,
            std::nullopt,
            std::nullopt,
            std::nullopt}),
       makeNullableFlatVector<int32_t>(
           {1, 2, std::nullopt, 4, 5, 6, std::nullopt, 7, 8, 9}),
       makeNullableFlatVector<int64_t>({1, 2, 3, 4, 1, 2, 1, 1, 2, 3})});
  assertQuery(op, expected);

  // Test with array wrapped in dictionary.
  auto reversedIndices = makeIndicesInReverse(6);
  auto vectorInDict = makeRowVector(
      {makeNullableFlatVector<double>({1.1, 2.2, 3.3, 4.4, 5.5, std::nullopt}),
       wrapInDictionary(reversedIndices, 6, array)});
  op = PlanBuilder()
           .values({vectorInDict})
           .unnest({"c0"}, {"c1"}, "ordinal")
           .planNode();

  auto expectedInDict = makeRowVector(
      {makeNullableFlatVector<double>(
           {1.1,
            1.1,
            1.1,
            2.2,
            4.4,
            4.4,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt}),
       makeNullableFlatVector<int32_t>(
           {7, 8, 9, std::nullopt, 5, 6, 1, 2, std::nullopt, 4}),
       makeNullableFlatVector<int64_t>({1, 2, 3, 1, 1, 2, 1, 2, 3, 4})});
  assertQuery(op, expectedInDict);
}

TEST_F(UnnestTest, arrayOfRows) {
  // Array of rows with legacy unnest set to false.
  auto rowType1 = ROW({INTEGER(), VARCHAR()});
  auto rowType2 = ROW({BIGINT(), DOUBLE(), VARCHAR()});
  std::vector<std::vector<std::optional<std::tuple<int32_t, std::string>>>>
      array1{
          {
              {{1, "red"}},
              {{2, "blue"}},
          },
          {
              {{3, "green"}},
              {{4, "orange"}},
              {{5, "black"}},
              {{6, "white"}},
          },
      };
  std::vector<
      std::vector<std::optional<std::tuple<int64_t, double, std::string>>>>
      array2{
          {
              {{-1, 1.0, "cat"}},
              {{-2, 2.0, "tiger"}},
              {{-3, 3.0, "lion"}},
          },
          {
              {{-4, 4.0, "elephant"}},
              {{-5, 5.0, "wolf"}},
          },
      };

  auto vector = makeRowVector(
      {makeArrayOfRowVector(array1, rowType1),
       makeArrayOfRowVector(array2, rowType2)});

  auto op = PlanBuilder()
                .values({vector})
                .unnest({}, {"c0", "c1"}, std::nullopt, true)
                .planNode();
  auto expected = makeRowVector(
      {makeNullableFlatVector<int32_t>({1, 2, std::nullopt, 3, 4, 5, 6}),
       makeNullableFlatVector<StringView>(
           {"red", "blue", std::nullopt, "green", "orange", "black", "white"}),
       makeNullableFlatVector<int64_t>(
           {-1, -2, -3, -4, -5, std::nullopt, std::nullopt}),
       makeNullableFlatVector<double>(
           {1.0, 2.0, 3.0, 4.0, 5.0, std::nullopt, std::nullopt}),
       makeNullableFlatVector<StringView>(
           {"cat",
            "tiger",
            "lion",
            "elephant",
            "wolf",
            std::nullopt,
            std::nullopt})});
  assertQuery(op, expected);

  // Test with array of rows (one of which is a constant). Each array is
  // encoded.
  vector_size_t size = array1.size();
  auto offsets = makeIndicesInReverse(size);
  vector = makeRowVector(
      {wrapInDictionary(offsets, size, makeArrayOfRowVector(array1, rowType1)),
       BaseVector::wrapInConstant(
           size, 0, makeArrayOfRowVector(array2, rowType2))});

  op = PlanBuilder()
           .values({vector})
           .unnest({}, {"c0", "c1"}, std::nullopt, true)
           .planNode();
  expected = makeRowVector(
      {makeNullableFlatVector<int32_t>({3, 4, 5, 6, 1, 2, std::nullopt}),
       makeNullableFlatVector<StringView>({
           "green",
           "orange",
           "black",
           "white",
           "red",
           "blue",
           std::nullopt,
       }),
       makeNullableFlatVector<int64_t>({-1, -2, -3, std::nullopt, -1, -2, -3}),
       makeNullableFlatVector<double>(
           {1.0, 2.0, 3.0, std::nullopt, 1.0, 2.0, 3.0}),
       makeNullableFlatVector<StringView>(
           {"cat", "tiger", "lion", std::nullopt, "cat", "tiger", "lion"})});
  assertQuery(op, expected);
}

TEST_F(UnnestTest, encodedArrayOfRows) {
  VectorFuzzer::Options opts;
  opts.containerVariableLength = false;
  opts.dictionaryHasNulls = false;
  VectorFuzzer fuzzer(opts, pool());

  auto unnestPlan = [&](const VectorPtr& input) -> const core::PlanNodePtr {
    return PlanBuilder()
        .values({makeRowVector({fuzzer.fuzzArray(input, 20)})})
        .unnest({}, {"c0"}, std::nullopt, true)
        .orderBy({"c0_c0_e", "c0_c1_e", "c0_c2_e"}, false)
        .planNode();
  };

  auto noUnnestPlan = [&](const VectorPtr& child1,
                          const VectorPtr& child2,
                          const VectorPtr& child3) -> const core::PlanNodePtr {
    return PlanBuilder()
        .values({makeRowVector({child1, child2, child3})})
        .orderBy({"c0", "c1", "c2"}, false)
        .planNode();
  };

  // The first test is to unnest an array of rows where the row vector is
  // encoded before building the array.
  auto arrayInput = fuzzer.fuzzDictionary(makeRowVector(
      {fuzzer.fuzzFlat(INTEGER(), 100),
       fuzzer.fuzzFlat(BIGINT(), 100),
       fuzzer.fuzzFlat(INTEGER(), 100)}));
  // Flatten the array vector to generate an equivalent plan without unnest.
  DecodedVector decoded;
  decoded.decode(*arrayInput);
  auto flatChildAt = [&](auto idx) -> const VectorPtr {
    auto decodedRowVector = decoded.base()->as<RowVector>();
    return decoded.wrap(
        decodedRowVector->childAt(idx), *arrayInput, decodedRowVector->size());
  };
  assertEqualResults(
      unnestPlan(arrayInput),
      noUnnestPlan(flatChildAt(0), flatChildAt(1), flatChildAt(2)));

  // The second unnest test is with an array of rows where each child row vector
  // is encoded before building the array row elements.
  auto child1 = fuzzer.fuzzDictionary(fuzzer.fuzzFlat(INTEGER(), 100));
  auto child2 = fuzzer.fuzzDictionary(fuzzer.fuzzFlat(BIGINT(), 100));
  auto child3 = fuzzer.fuzzDictionary(fuzzer.fuzzFlat(INTEGER(), 100));
  arrayInput = makeRowVector({child1, child2, child3});
  assertEqualResults(
      unnestPlan(arrayInput), noUnnestPlan(child1, child2, child3));
}

TEST_F(UnnestTest, basicMap) {
  auto vector = makeRowVector(
      {makeFlatVector<int64_t>(100, [](auto row) { return row; }),
       makeMapVector<int64_t, double>(
           100,
           [](auto /* row */) { return 2; },
           [](auto row) { return row % 2; },
           [](auto row) { return row % 2 + 1; })});
  auto op = PlanBuilder().values({vector}).unnest({"c0"}, {"c1"}).planNode();
  // DuckDB doesn't support Unnest from MAP column. Hence,using 2 separate array
  // columns with the keys and values part of the MAP to validate.
  auto duckDbVector = makeRowVector(
      {makeFlatVector<int64_t>(100, [](auto row) { return row; }),
       makeArrayVector<int32_t>(
           100,
           [](auto /* row */) { return 2; },
           [](auto /* row */, auto index) { return index; }),
       makeArrayVector<int32_t>(
           100,
           [](auto /* row */) { return 2; },
           [](auto /* row */, auto index) { return index + 1; })});
  createDuckDbTable({duckDbVector});
  assertQuery(op, "SELECT c0, UNNEST(c1), UNNEST(c2) FROM tmp");
}

TEST_F(UnnestTest, mapWithOrdinality) {
  auto map = makeMapVector<int32_t, double>(
      {{{1, 1.1}, {2, std::nullopt}},
       {{3, 3.3}, {4, 4.4}, {5, 5.5}},
       {{6, std::nullopt}}});
  auto vector =
      makeRowVector({makeNullableFlatVector<int32_t>({1, 2, 3}), map});

  auto op = PlanBuilder()
                .values({vector})
                .unnest({"c0"}, {"c1"}, "ordinal")
                .planNode();

  auto expected = makeRowVector(
      {makeNullableFlatVector<int32_t>({1, 1, 2, 2, 2, 3}),
       makeNullableFlatVector<int32_t>({1, 2, 3, 4, 5, 6}),
       makeNullableFlatVector<double>(
           {1.1, std::nullopt, 3.3, 4.4, 5.5, std::nullopt}),
       makeNullableFlatVector<int64_t>({1, 2, 1, 2, 3, 1})});
  assertQuery(op, expected);

  // Test with map wrapped in dictionary.
  auto reversedIndices = makeIndicesInReverse(3);
  auto vectorInDict = makeRowVector(
      {makeNullableFlatVector<int32_t>({1, 2, 3}),
       wrapInDictionary(reversedIndices, 3, map)});
  op = PlanBuilder()
           .values({vectorInDict})
           .unnest({"c0"}, {"c1"}, "ordinal")
           .planNode();

  auto expectedInDict = makeRowVector(
      {makeNullableFlatVector<int32_t>({1, 2, 2, 2, 3, 3}),
       makeNullableFlatVector<int32_t>({6, 3, 4, 5, 1, 2}),
       makeNullableFlatVector<double>(
           {std::nullopt, 3.3, 4.4, 5.5, 1.1, std::nullopt}),
       makeNullableFlatVector<int64_t>({1, 1, 2, 3, 1, 2})});
  assertQuery(op, expectedInDict);
}

TEST_F(UnnestTest, multipleColumns) {
  std::vector<vector_size_t> offsets(100, 0);
  for (int i = 1; i < 100; ++i) {
    offsets[i] = offsets[i - 1] + i % 11 + 1;
  }

  auto vector = makeRowVector(
      {makeFlatVector<int64_t>(100, [](auto row) { return row; }),
       vectorMaker_.mapVector<int64_t, double>(
           100,
           [](auto row) { return row % 7 + 1; },
           [](auto row, auto index) { return index * (row % 4); },
           [](auto row, auto index) { return index * (row % 4) + 1; },
           nullEvery(6)),
       makeArrayVector<int32_t>(
           100,
           [](auto row) { return row % 5 + 1; },
           [](auto row, auto index) { return index * (row % 3); },
           nullEvery(7)),
       makeArrayVector(offsets, makeConstant<int32_t>(7, 700))});

  auto op = PlanBuilder()
                .values({vector})
                .unnest({"c0"}, {"c1", "c2", "c3"})
                .planNode();

  // DuckDB doesn't support Unnest from MAP column. Hence,using 2 separate array
  // columns with the keys and values part of the MAP to validate.
  auto duckDbVector = makeRowVector(
      {makeFlatVector<int64_t>(100, [](auto row) { return row; }),
       makeArrayVector<int64_t>(
           100,
           [](auto row) { return row % 7 + 1; },
           [](auto row, auto index) { return index * (row % 4); },
           nullEvery(6)),
       makeArrayVector<double>(
           100,
           [](auto row) { return row % 7 + 1; },
           [](auto row, auto index) { return index * (row % 4) + 1; },
           nullEvery(6)),
       makeArrayVector<int32_t>(
           100,
           [](auto row) { return row % 5 + 1; },
           [](auto row, auto index) { return index * (row % 3); },
           nullEvery(7)),
       makeArrayVector(offsets, makeConstant<int32_t>(7, 700))});
  createDuckDbTable({duckDbVector});
  assertQuery(
      op, "SELECT c0, UNNEST(c1), UNNEST(c2), UNNEST(c3), UNNEST(c4) FROM tmp");
}

TEST_F(UnnestTest, multipleColumnsWithOrdinality) {
  std::vector<vector_size_t> offsets(100, 0);
  for (int i = 1; i < 100; ++i) {
    offsets[i] = offsets[i - 1] + i % 11 + 1;
  }

  auto vector = makeRowVector(
      {makeFlatVector<int64_t>(100, [](auto row) { return row; }),
       vectorMaker_.mapVector<int64_t, double>(
           100,
           [](auto row) { return row % 7 + 1; },
           [](auto row, auto index) { return index * (row % 4); },
           [](auto row, auto index) { return index * (row % 4) + 1; },
           nullEvery(6)),
       makeArrayVector<int32_t>(
           100,
           [](auto row) { return row % 5 + 1; },
           [](auto row, auto index) { return index * (row % 3); },
           nullEvery(7)),
       makeArrayVector(offsets, makeConstant<int32_t>(7, 700))});

  auto op = PlanBuilder()
                .values({vector})
                .unnest({"c0"}, {"c1", "c2", "c3"}, "ordinal")
                .planNode();

  // DuckDB doesn't support Unnest from MAP column. Hence,using 2 separate array
  // columns with the keys and values part of the MAP to validate.
  auto ordinalitySize = [&](auto row) {
    if (row % 42 == 0) {
      return offsets[row + 1] - offsets[row];
    } else if (row % 7 == 0) {
      return std::max(row % 7 + 1, offsets[row + 1] - offsets[row]);
    } else if (row % 6 == 0) {
      return std::max(row % 5 + 1, offsets[row + 1] - offsets[row]);
    } else {
      return std::max(
          std::max(row % 5, row % 7) + 1,
          (row == 99 ? 700 : offsets[row + 1]) - offsets[row]);
    }
  };

  auto duckDbVector = makeRowVector(
      {makeFlatVector<int64_t>(100, [](auto row) { return row; }),
       makeArrayVector<int64_t>(
           100,
           [](auto row) { return row % 7 + 1; },
           [](auto row, auto index) { return index * (row % 4); },
           nullEvery(6)),
       makeArrayVector<double>(
           100,
           [](auto row) { return row % 7 + 1; },
           [](auto row, auto index) { return index * (row % 4) + 1; },
           nullEvery(6)),
       makeArrayVector<int32_t>(
           100,
           [](auto row) { return row % 5 + 1; },
           [](auto row, auto index) { return index * (row % 3); },
           nullEvery(7)),
       makeArrayVector(offsets, makeConstant<int32_t>(7, 700)),
       makeArrayVector<int64_t>(
           100, ordinalitySize, [](auto /* row */, auto index) {
             return index + 1;
           })});
  createDuckDbTable({duckDbVector});
  assertQuery(
      op,
      "SELECT c0, UNNEST(c1), UNNEST(c2), UNNEST(c3), UNNEST(c4), UNNEST(c5) FROM tmp");

  // Test with empty arrays and maps.
  vector = makeRowVector(
      {makeNullableFlatVector<double>({1.1, 2.2, std::nullopt, 4.4, 5.5}),
       vectorMaker_.arrayVectorNullable<int32_t>(
           {{{1, 2, std::nullopt, 4}}, std::nullopt, {{5, 6}}, {}, {{7}}}),
       makeMapVector<int32_t, double>(
           {{{1, 1.1}, {2, std::nullopt}},
            {{3, 3.3}, {4, 4.4}, {5, 5.5}},
            {{6, std::nullopt}},
            {},
            {}})});

  op = PlanBuilder()
           .values({vector})
           .unnest({"c0"}, {"c1", "c2"}, "ordinal")
           .planNode();

  auto expected = makeRowVector(
      {makeNullableFlatVector<double>(
           {1.1,
            1.1,
            1.1,
            1.1,
            2.2,
            2.2,
            2.2,
            std::nullopt,
            std::nullopt,
            5.5}),
       makeNullableFlatVector<int32_t>(
           {1,
            2,
            std::nullopt,
            4,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            5,
            6,
            7}),
       makeNullableFlatVector<int32_t>(
           {1,
            2,
            std::nullopt,
            std::nullopt,
            3,
            4,
            5,
            6,
            std::nullopt,
            std::nullopt}),
       makeNullableFlatVector<double>(
           {1.1,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            3.3,
            4.4,
            5.5,
            std::nullopt,
            std::nullopt,
            std::nullopt}),
       makeNullableFlatVector<int64_t>({1, 2, 3, 4, 1, 2, 3, 1, 2, 1})});
  assertQuery(op, expected);
}

TEST_F(UnnestTest, allEmptyOrNullArrays) {
  auto vector = makeRowVector(
      {makeFlatVector<int64_t>(100, [](auto row) { return row; }),
       makeArrayVector<int32_t>(
           100,
           [](auto /* row */) { return 0; },
           [](auto /* row */, auto index) { return index; },
           nullEvery(5)),
       makeArrayVector<int32_t>(
           100,
           [](auto /* row */) { return 0; },
           [](auto /* row */, auto index) { return index; },
           nullEvery(7))});

  auto op =
      PlanBuilder().values({vector}).unnest({"c0"}, {"c1", "c2"}).planNode();
  assertQueryReturnsEmptyResult(op);

  op = PlanBuilder()
           .values({vector})
           .unnest({"c0"}, {"c1", "c2"}, "ordinal")
           .planNode();
  assertQueryReturnsEmptyResult(op);
}

TEST_F(UnnestTest, allEmptyOrNullMaps) {
  auto vector = makeRowVector(
      {makeFlatVector<int64_t>(100, [](auto row) { return row; }),
       makeMapVector<int64_t, double>(
           100,
           [](auto /* row */) { return 0; },
           [](auto /* row */) { return 0; },
           [](auto /* row */) { return 0; },
           nullEvery(5)),
       makeMapVector<int64_t, double>(
           100,
           [](auto /* row */) { return 0; },
           [](auto /* row */) { return 0; },
           [](auto /* row */) { return 0; },
           nullEvery(7))});

  auto op =
      PlanBuilder().values({vector}).unnest({"c0"}, {"c1", "c2"}).planNode();
  assertQueryReturnsEmptyResult(op);

  op = PlanBuilder()
           .values({vector})
           .unnest({"c0"}, {"c1", "c2"}, "ordinal")
           .planNode();
  assertQueryReturnsEmptyResult(op);
}

TEST_F(UnnestTest, batchSize) {
  auto data = makeRowVector({
      makeFlatVector<int64_t>(10'000, [](auto row) { return row; }),
  });

  // Unnest 10K rows into 30K rows.
  core::PlanNodeId unnestId;
  auto plan = PlanBuilder()
                  .values({data})
                  .project({"sequence(1, 3) as s"})
                  .unnest({}, {"s"})
                  .capturePlanNodeId(unnestId)
                  .planNode();

  auto expected = makeRowVector({
      makeFlatVector<int64_t>(10'000 * 3, [](auto row) { return 1 + row % 3; }),
  });

  // 17 rows per output allows to unnest 6 input rows at a time.
  {
    auto task = AssertQueryBuilder(plan)
                    .config(core::QueryConfig::kPreferredOutputBatchRows, "17")
                    .assertResults({expected});
    auto stats = exec::toPlanStats(task->taskStats());

    ASSERT_EQ(30'000, stats.at(unnestId).outputRows);
    ASSERT_EQ(1 + 10'000 / 6, stats.at(unnestId).outputVectors);
  }

  // 2 rows per output allows to unnest 1 input row at a time.
  {
    auto task = AssertQueryBuilder(plan)
                    .config(core::QueryConfig::kPreferredOutputBatchRows, "2")
                    .assertResults({expected});
    auto stats = exec::toPlanStats(task->taskStats());

    ASSERT_EQ(30'000, stats.at(unnestId).outputRows);
    ASSERT_EQ(10'000, stats.at(unnestId).outputVectors);
  }

  // 100K rows per output allows to unnest all at once.
  {
    auto task =
        AssertQueryBuilder(plan)
            .config(core::QueryConfig::kPreferredOutputBatchRows, "100000")
            .assertResults({expected});
    auto stats = exec::toPlanStats(task->taskStats());

    ASSERT_EQ(30'000, stats.at(unnestId).outputRows);
    ASSERT_EQ(1, stats.at(unnestId).outputVectors);
  }
}
