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

#include "velox/dwio/common/tests/utils/BatchMaker.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/QueryAssertions.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec::test;

class AggregationDistinctTest : public OperatorTestBase {
 public:
  template <class NativeType>
  std::function<VectorPtr()> baseFunctorTemplate() {
    return [&]() {
      return makeFlatVector<NativeType>(
          2, [&](vector_size_t row) { return (NativeType)(row % 2); }, nullptr);
    };
  }

// This test verifies this query:
// WITH sample AS ( SELECT * FROM (VALUES (1,1,1), (1,1,2), (1,1,100), (2,1,2),
// (2,2,1), (2,3,2)) AS account (c0, c1, c2) ) SELECT c0,sum(distinct
// c1),sum(distinct c2) from sample group by c0;
TEST_F(AggregationDistinctTest, small11) {
  std::vector<RowVectorPtr> vectors;
  // Simulate the input over 3 splits.
  vectors.push_back(makeRowVector(
      {makeFlatVector<int32_t>({1, 1}),
       makeFlatVector<int64_t>({1, 1}),
       makeFlatVector<int8_t>({1, 2})}));
  vectors.push_back(makeRowVector(
      {makeFlatVector<int32_t>({1, 2}),
       makeFlatVector<int64_t>({1, 1}),
       makeFlatVector<int8_t>({100, 2})}));
  vectors.push_back(makeRowVector(
      {makeFlatVector<int32_t>({2, 2}),
       makeFlatVector<int64_t>({2, 3}),
       makeFlatVector<int8_t>({1, 2})}));

  auto op =
      PlanBuilder()
          .values(vectors)
          .singleAggregation({"c0"}, {"sum(c1)", "sum(c2)"}, {}, {true, true})
          .orderBy({"c0"}, false)
          .planNode();

  CursorParameters params;
  params.planNode = op;

  auto result = readCursor(params, [](auto) {});
  auto actual = result.second;

  RowVectorPtr expected = makeRowVector(
      {makeFlatVector<int32_t>({1, 2}),
       makeFlatVector<int64_t>({1, 6}),
       makeFlatVector<int64_t>({103, 3})});
  assertEqualVectors(expected, actual[0]);
}

// This test verifies this query:
// WITH sample AS ( SELECT * FROM (VALUES (1,1,1), (1,1,2), (1,1,100), (2,1,2),
// (2,2,1), (2,3,2)) AS account (c0, c1, c2) ) SELECT c0,sum(c1),sum(distinct
// c2) from sample group by c0;
TEST_F(AggregationDistinctTest, small01) {
  std::vector<RowVectorPtr> vectors;
  // Simulate the input over 3 splits.
  vectors.push_back(makeRowVector(
      {makeFlatVector<int32_t>({1, 1}),
       makeFlatVector<int64_t>({1, 1}),
       makeFlatVector<int8_t>({1, 2})}));
  vectors.push_back(makeRowVector(
      {makeFlatVector<int32_t>({1, 2}),
       makeFlatVector<int64_t>({1, 1}),
       makeFlatVector<int8_t>({100, 2})}));
  vectors.push_back(makeRowVector(
      {makeFlatVector<int32_t>({2, 2}),
       makeFlatVector<int64_t>({2, 3}),
       makeFlatVector<int8_t>({1, 2})}));

  auto op =
      PlanBuilder()
          .values(vectors)
          .singleAggregation({"c0"}, {"sum(c1)", "sum(c2)"}, {}, {false, true})
          .orderBy({"c0"}, false)
          .planNode();

  CursorParameters params;
  params.planNode = op;

  auto result = readCursor(params, [](auto) {});
  auto actual = result.second;

  RowVectorPtr expected = makeRowVector(
      {makeFlatVector<int32_t>({1, 2}),
       makeFlatVector<int64_t>({3, 6}),
       makeFlatVector<int64_t>({103, 3})});
  assertEqualVectors(expected, actual[0]);
}

// This test verifies this query:
// WITH sample AS ( SELECT * FROM (VALUES (1,1,1), (1,1,2), (1,1,100), (2,1,2),
// (2,2,1), (2,3,2)) AS account (c0, c1, c2) ) SELECT c0,sum(distinct
// c1),sum(c2) from sample group by c0;
TEST_F(AggregationDistinctTest, small10) {
  std::vector<RowVectorPtr> vectors;
  // Simulate the input over 3 splits.
  vectors.push_back(makeRowVector(
      {makeFlatVector<int32_t>({1, 1}),
       makeFlatVector<int16_t>({1, 1}),
       makeFlatVector<int32_t>({1, 2})}));
  vectors.push_back(makeRowVector(
      {makeFlatVector<int32_t>({1, 2}),
       makeFlatVector<int16_t>({1, 1}),
       makeFlatVector<int32_t>({100, 2})}));
  vectors.push_back(makeRowVector(
      {makeFlatVector<int32_t>({2, 2}),
       makeFlatVector<int16_t>({2, 3}),
       makeFlatVector<int32_t>({1, 2})}));

  auto op =
      PlanBuilder()
          .values(vectors)
          .singleAggregation({"c0"}, {"sum(c1)", "sum(c2)"}, {}, {true, false})
          .orderBy({"c0"}, false)
          .planNode();

  CursorParameters params;
  params.planNode = op;

  auto result = readCursor(params, [](auto) {});
  auto actual = result.second;

  RowVectorPtr expected = makeRowVector(
      {makeFlatVector<int32_t>({1, 2}),
       makeFlatVector<int64_t>({1, 6}),
       makeFlatVector<int64_t>({103, 5})});
  assertEqualVectors(expected, actual[0]);
}

// This test verifies this query:
// WITH sample AS ( SELECT * FROM (VALUES (1,1,1), (1,1,2), (1,1,100), (2,1,2),
// (2,2,1), (2,3,2)) AS account (c0, c1, c2) ) SELECT c0,sum(c1),sum(c2) from
// sample group by c0;
TEST_F(AggregationDistinctTest, small00) {
  std::vector<RowVectorPtr> vectors;
  // Simulate the input over 3 splits.
  vectors.push_back(makeRowVector(
      {makeFlatVector<int32_t>({1, 1}),
       makeFlatVector<int16_t>({1, 1}),
       makeFlatVector<int32_t>({1, 2})}));
  vectors.push_back(makeRowVector(
      {makeFlatVector<int32_t>({1, 2}),
       makeFlatVector<int16_t>({1, 1}),
       makeFlatVector<int32_t>({100, 2})}));
  vectors.push_back(makeRowVector(
      {makeFlatVector<int32_t>({2, 2}),
       makeFlatVector<int16_t>({2, 3}),
       makeFlatVector<int32_t>({1, 2})}));

  auto op =
      PlanBuilder()
          .values(vectors)
          .singleAggregation({"c0"}, {"sum(c1)", "sum(c2)"}, {}, {false, false})
          .orderBy({"c0"}, false)
          .planNode();

  CursorParameters params;
  params.planNode = op;

  auto result = readCursor(params, [](auto) {});
  auto actual = result.second;

  RowVectorPtr expected = makeRowVector(
      {makeFlatVector<int32_t>({1, 2}),
       makeFlatVector<int64_t>({3, 6}),
       makeFlatVector<int64_t>({103, 5})});
  assertEqualVectors(expected, actual[0]);
}

// This test verifies this query:
// WITH sample AS ( SELECT * FROM (VALUES (1,1,1), (1,1,2), (1,1,100), (1,2,2),
// (2,2,1), (3,2,2)) AS account (c0, c1, c2) ) SELECT c1,sum(distinct
// c0),sum(distinct c2) from sample group by c1;
TEST_F(AggregationDistinctTest, small11Shuffled) {
  std::vector<RowVectorPtr> vectors;
  // Simulate the input over 3 splits.
  vectors.push_back(makeRowVector(
      {makeFlatVector<int64_t>({1, 1}),
       makeFlatVector<int32_t>({1, 1}),
       makeFlatVector<int8_t>({1, 2})}));
  vectors.push_back(makeRowVector(
      {makeFlatVector<int64_t>({1, 1}),
       makeFlatVector<int32_t>({1, 2}),
       makeFlatVector<int8_t>({100, 2})}));
  vectors.push_back(makeRowVector(
      {makeFlatVector<int64_t>({2, 3}),
       makeFlatVector<int32_t>({2, 2}),
       makeFlatVector<int8_t>({1, 2})}));

  auto op =
      PlanBuilder()
          .values(vectors)
          .singleAggregation({"c1"}, {"sum(c0)", "sum(c2)"}, {}, {true, true})
          .orderBy({"c1"}, false)
          .planNode();

  CursorParameters params;
  params.planNode = op;

  auto result = readCursor(params, [](auto) {});
  auto actual = result.second;

  RowVectorPtr expected = makeRowVector(
      {makeFlatVector<int32_t>({1, 2}),
       makeFlatVector<int64_t>({1, 6}),
       makeFlatVector<int64_t>({103, 3})});
  assertEqualVectors(expected, actual[0]);
}

// This test verifies this query:
// WITH sample AS ( SELECT * FROM (VALUES (1,1,2,3), (1,1,3,4), (2,11,12,13),
// (3,21,22,23), (1,2,3,3), (2,11,13,13), (2,12,12,15), (3,21,21,21), (1,2,2,5),
// (2,13,14,15), (3,21,23,23), (3,22,23,21)) AS account (c0, c1, c2, c3) )
// SELECT c0,sum(distinct c1),sum(distinct c2),sum(distinct c3) from sample
// group by c0;
TEST_F(AggregationDistinctTest, big111) {
  std::vector<RowVectorPtr> vectors;
  // Simulate the input over 3 splits.
  vectors.push_back(makeRowVector(
      {makeFlatVector<int32_t>({1, 1, 2, 3}),
       makeFlatVector<int16_t>({1, 1, 11, 21}),
       makeFlatVector<int32_t>({2, 3, 12, 22}),
       makeFlatVector<int32_t>({3, 4, 13, 23})}));
  vectors.push_back(makeRowVector(
      {makeFlatVector<int32_t>({1, 2, 2, 3}),
       makeFlatVector<int16_t>({2, 11, 12, 21}),
       makeFlatVector<int32_t>({3, 13, 12, 21}),
       makeFlatVector<int32_t>({3, 13, 15, 21})}));
  vectors.push_back(makeRowVector(
      {makeFlatVector<int32_t>({1, 2, 3, 3}),
       makeFlatVector<int16_t>({2, 13, 21, 22}),
       makeFlatVector<int32_t>({2, 14, 23, 23}),
       makeFlatVector<int32_t>({5, 15, 23, 21})}));

  auto op =
      PlanBuilder()
          .values(vectors)
          .singleAggregation(
              {"c0"}, {"sum(c1)", "sum(c2)", "sum(c3)"}, {}, {true, true, true})
          .orderBy({"c0"}, false)
          .planNode();

  CursorParameters params;
  params.planNode = op;

  auto result = readCursor(params, [](auto) {});
  auto actual = result.second;

  RowVectorPtr expected = makeRowVector(
      {makeFlatVector<int32_t>({1, 2, 3}),
       makeFlatVector<int64_t>({3, 36, 43}),
       makeFlatVector<int64_t>({
           5,
           39,
           66,
       }),
       makeFlatVector<int64_t>({12, 28, 44})});
  assertEqualVectors(expected, actual[0]);
}

// This test verifies this query:
// WITH sample AS ( SELECT * FROM (VALUES (1,1,2,3), (1,1,3,4), (2,11,12,13),
// (3,21,22,23), (1,2,3,3), (2,11,13,13), (2,12,12,15), (3,21,21,21), (1,2,2,5),
// (2,13,14,15), (3,21,23,23), (3,22,23,21)) AS account (c0, c1, c2, c3) )
// SELECT c0,sum(c1),sum(distinct c2),sum(distinct c3) from sample
// group by c0;
TEST_F(AggregationDistinctTest, big011) {
  std::vector<RowVectorPtr> vectors;
  // Simulate the input over 3 splits.
  vectors.push_back(makeRowVector(
      {makeFlatVector<int32_t>({1, 1, 2, 3}),
       makeFlatVector<int16_t>({1, 1, 11, 21}),
       makeFlatVector<int32_t>({2, 3, 12, 22}),
       makeFlatVector<int32_t>({3, 4, 13, 23})}));
  vectors.push_back(makeRowVector(
      {makeFlatVector<int32_t>({1, 2, 2, 3}),
       makeFlatVector<int16_t>({2, 11, 12, 21}),
       makeFlatVector<int32_t>({3, 13, 12, 21}),
       makeFlatVector<int32_t>({3, 13, 15, 21})}));
  vectors.push_back(makeRowVector(
      {makeFlatVector<int32_t>({1, 2, 3, 3}),
       makeFlatVector<int16_t>({2, 13, 21, 22}),
       makeFlatVector<int32_t>({2, 14, 23, 23}),
       makeFlatVector<int32_t>({5, 15, 23, 21})}));

  auto op = PlanBuilder()
                .values(vectors)
                .singleAggregation(
                    {"c0"},
                    {"sum(c1)", "sum(c2)", "sum(c3)"},
                    {},
                    {false, true, true})
                .orderBy({"c0"}, false)
                .planNode();

  CursorParameters params;
  params.planNode = op;

  auto result = readCursor(params, [](auto) {});
  auto actual = result.second;

  RowVectorPtr expected = makeRowVector(
      {makeFlatVector<int32_t>({1, 2, 3}),
       makeFlatVector<int64_t>({6, 47, 85}),
       makeFlatVector<int64_t>({5, 39, 66}),
       makeFlatVector<int64_t>({12, 28, 44})});
  assertEqualVectors(expected, actual[0]);
}

// This test verifies this query:
// WITH sample AS ( SELECT * FROM (VALUES (1,1,2,3), (1,1,3,4), (2,11,12,13),
// (3,21,22,23), (1,2,3,3), (2,11,13,13), (2,12,12,15), (3,21,21,21), (1,2,2,5),
// (2,13,14,15), (3,21,23,23), (3,22,23,21)) AS account (c0, c1, c2, c3) )
// SELECT c0,sum(distinct c1),sum(c2),sum(distinct c3) from sample
// group by c0;
TEST_F(AggregationDistinctTest, big101) {
  std::vector<RowVectorPtr> vectors;
  // Simulate the input over 3 splits.
  vectors.push_back(makeRowVector(
      {makeFlatVector<int32_t>({1, 1, 2, 3}),
       makeFlatVector<int16_t>({1, 1, 11, 21}),
       makeFlatVector<int32_t>({2, 3, 12, 22}),
       makeFlatVector<int32_t>({3, 4, 13, 23})}));
  vectors.push_back(makeRowVector(
      {makeFlatVector<int32_t>({1, 2, 2, 3}),
       makeFlatVector<int16_t>({2, 11, 12, 21}),
       makeFlatVector<int32_t>({3, 13, 12, 21}),
       makeFlatVector<int32_t>({3, 13, 15, 21})}));
  vectors.push_back(makeRowVector(
      {makeFlatVector<int32_t>({1, 2, 3, 3}),
       makeFlatVector<int16_t>({2, 13, 21, 22}),
       makeFlatVector<int32_t>({2, 14, 23, 23}),
       makeFlatVector<int32_t>({5, 15, 23, 21})}));

  auto op = PlanBuilder()
                .values(vectors)
                .singleAggregation(
                    {"c0"},
                    {"sum(c1)", "sum(c2)", "sum(c3)"},
                    {},
                    {true, false, true})
                .orderBy({"c0"}, false)
                .planNode();

  CursorParameters params;
  params.planNode = op;

  auto result = readCursor(params, [](auto) {});
  auto actual = result.second;

  RowVectorPtr expected = makeRowVector(
      {makeFlatVector<int32_t>({1, 2, 3}),
       makeFlatVector<int64_t>({3, 36, 43}),
       makeFlatVector<int64_t>({10, 51, 89}),
       makeFlatVector<int64_t>({12, 28, 44})});
  assertEqualVectors(expected, actual[0]);
}

// This test verifies this query:
// WITH sample AS ( SELECT * FROM (VALUES (1,1,2,3), (1,1,3,4), (2,11,12,13),
// (3,21,22,23), (1,2,3,3), (2,11,13,13), (2,12,12,15), (3,21,21,21), (1,2,2,5),
// (2,13,14,15), (3,21,23,23), (3,22,23,21)) AS account (c0, c1, c2, c3) )
// SELECT c0,sum(distinct c1),sum(distinct c2),sum(c3) from sample
// group by c0;
TEST_F(AggregationDistinctTest, big110) {
  std::vector<RowVectorPtr> vectors;
  // Simulate the input over 3 splits.
  vectors.push_back(makeRowVector(
      {makeFlatVector<int32_t>({1, 1, 2, 3}),
       makeFlatVector<int16_t>({1, 1, 11, 21}),
       makeFlatVector<int32_t>({2, 3, 12, 22}),
       makeFlatVector<int32_t>({3, 4, 13, 23})}));
  vectors.push_back(makeRowVector(
      {makeFlatVector<int32_t>({1, 2, 2, 3}),
       makeFlatVector<int16_t>({2, 11, 12, 21}),
       makeFlatVector<int32_t>({3, 13, 12, 21}),
       makeFlatVector<int32_t>({3, 13, 15, 21})}));
  vectors.push_back(makeRowVector(
      {makeFlatVector<int32_t>({1, 2, 3, 3}),
       makeFlatVector<int16_t>({2, 13, 21, 22}),
       makeFlatVector<int32_t>({2, 14, 23, 23}),
       makeFlatVector<int32_t>({5, 15, 23, 21})}));

  auto op = PlanBuilder()
                .values(vectors)
                .singleAggregation(
                    {"c0"},
                    {"sum(c1)", "sum(c2)", "sum(c3)"},
                    {},
                    {true, true, false})
                .orderBy({"c0"}, false)
                .planNode();

  CursorParameters params;
  params.planNode = op;

  auto result = readCursor(params, [](auto) {});
  auto actual = result.second;

  RowVectorPtr expected = makeRowVector(
      {makeFlatVector<int32_t>({1, 2, 3}),
       makeFlatVector<int64_t>({3, 36, 43}),
       makeFlatVector<int64_t>({5, 39, 66}),
       makeFlatVector<int64_t>({15, 56, 88})});
  assertEqualVectors(expected, actual[0]);
}

// This test verifies this query:
// WITH sample AS ( SELECT * FROM (VALUES (1,1,2,3), (1,1,3,4), (2,11,12,13),
// (3,21,22,23), (1,2,3,3), (2,11,13,13), (2,12,12,15), (3,21,21,21), (1,2,2,5),
// (2,13,14,15), (3,21,23,23), (3,22,23,21)) AS account (c0, c1, c2, c3) )
// SELECT c0,sum(c1),sum(c2),sum(c3) from sample
// group by c0;
TEST_F(AggregationDistinctTest, big000) {
  std::vector<RowVectorPtr> vectors;
  // Simulate the input over 3 splits.
  vectors.push_back(makeRowVector(
      {makeFlatVector<int32_t>({1, 1, 2, 3}),
       makeFlatVector<int16_t>({1, 1, 11, 21}),
       makeFlatVector<int32_t>({2, 3, 12, 22}),
       makeFlatVector<int32_t>({3, 4, 13, 23})}));
  vectors.push_back(makeRowVector(
      {makeFlatVector<int32_t>({1, 2, 2, 3}),
       makeFlatVector<int16_t>({2, 11, 12, 21}),
       makeFlatVector<int32_t>({3, 13, 12, 21}),
       makeFlatVector<int32_t>({3, 13, 15, 21})}));
  vectors.push_back(makeRowVector(
      {makeFlatVector<int32_t>({1, 2, 3, 3}),
       makeFlatVector<int16_t>({2, 13, 21, 22}),
       makeFlatVector<int32_t>({2, 14, 23, 23}),
       makeFlatVector<int32_t>({5, 15, 23, 21})}));

  auto op = PlanBuilder()
                .values(vectors)
                .singleAggregation(
                    {"c0"},
                    {"sum(c1)", "sum(c2)", "sum(c3)"},
                    {},
                    {false, false, false})
                .orderBy({"c0"}, false)
                .planNode();

  CursorParameters params;
  params.planNode = op;

  auto result = readCursor(params, [](auto) {});
  auto actual = result.second;

  RowVectorPtr expected = makeRowVector(
      {makeFlatVector<int32_t>({1, 2, 3}),
       makeFlatVector<int64_t>({6, 47, 85}),
       makeFlatVector<int64_t>({10, 51, 89}),
       makeFlatVector<int64_t>({15, 56, 88})});
  assertEqualVectors(expected, actual[0]);
}
