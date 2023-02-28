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

#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec::test;

class MarkDistinctTest : public OperatorTestBase {
 public:
  void runBasicMarkDistinctTest(const VectorPtr& base) {
    std::vector<RowVectorPtr> vectors;

    const vector_size_t baseSize = base->size();
    const vector_size_t size = baseSize * 2;
    auto indices = makeIndices(size, [&](auto row) { return row % baseSize; });
    auto baseEncoded = wrapInDictionary(indices, size, base);

    vectors.push_back(makeRowVector({baseEncoded}));

    auto distinctCol =
        makeFlatVector<bool>(size, [&](auto row) { return row < baseSize; });

    auto expectedResults = makeRowVector({baseEncoded, distinctCol});

    auto op = PlanBuilder()
                  .values(vectors)
                  .markDistinct("c0$Distinct", {"c0"})
                  .planNode();

    auto results = AssertQueryBuilder(op).copyResults(pool());
    assertEqualVectors(results, expectedResults);
  }
};

template <typename T>
class MarkDistinctPODTest : public MarkDistinctTest {};

using MyTypes =
    ::testing::Types<int8_t, int16_t, int32_t, int64_t, float, double, bool>;
TYPED_TEST_SUITE(MarkDistinctPODTest, MyTypes);

TYPED_TEST(MarkDistinctPODTest, basic) {
  auto data = VectorTestBase::makeFlatVector<TypeParam>(
      2, [&](auto row) { return row % 2; });

  MarkDistinctTest::runBasicMarkDistinctTest(data);
}

TEST_F(MarkDistinctTest, array) {
  auto base = makeArrayVector<int64_t>({
      {1, 2, 3, 4, 5},
      {1, 2, 3},
  });
  runBasicMarkDistinctTest(base);
}

TEST_F(MarkDistinctTest, map) {
  auto base = makeMapVector<int8_t, int32_t>(
      {{{1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}}, {{1, 1}, {1, 1}, {1, 1}}});
  runBasicMarkDistinctTest(base);
}

TEST_F(MarkDistinctTest, varchar) {
  auto base = makeFlatVector<StringView>({
      "{1, 2, 3, 4, 5}",
      "{1, 2, 3}",
  });
  runBasicMarkDistinctTest(base);
}

TEST_F(MarkDistinctTest, row) {
  auto base = makeRowVector({
      makeArrayVector<int64_t>({
          {1, 2, 3, 4, 5},
          {1, 2, 3},
      }),
      makeMapVector<int8_t, int32_t>({
          {{1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}},
          {{1, 1}, {1, 1}, {1, 1}},
      }),
  });
  runBasicMarkDistinctTest(base);
}

// This test verifies this query:
// WITH sample AS ( SELECT * FROM (VALUES (1,1,1), (1,1,2), (1,1,1), (2,1,2),
// (2,2,1), (2,3,2)) AS account (c1, c2, c3) ) SELECT c1,sum(distinct
// c2),sum(distinct c3) from sample group by c1;
TEST_F(MarkDistinctTest, distinctAggregationTest) {
  std::vector<RowVectorPtr> vectors;
  // Simulate the input over 3 splits.
  vectors.push_back(makeRowVector(
      {makeFlatVector<int32_t>({1, 1}),
       makeFlatVector<int32_t>({1, 1}),
       makeFlatVector<int32_t>({1, 2})}));
  vectors.push_back(makeRowVector(
      {makeFlatVector<int32_t>({1, 2}),
       makeFlatVector<int32_t>({1, 1}),
       makeFlatVector<int32_t>({1, 2})}));
  vectors.push_back(makeRowVector(
      {makeFlatVector<int32_t>({2, 2}),
       makeFlatVector<int32_t>({2, 3}),
       makeFlatVector<int32_t>({1, 2})}));

  auto op =
      PlanBuilder()
          .values(vectors)
          .markDistinct("c1$Distinct", {"c0", "c1"})
          .markDistinct("c2$Distinct", {"c0", "c2"})
          .singleAggregation(
              {"c0"}, {"sum(c1)", "sum(c2)"}, {"c1$Distinct", "c2$Distinct"})
          .orderBy({"c0"}, false)
          .planNode();

  auto expected = makeRowVector(
      {makeFlatVector<int32_t>({1, 2}),
       makeFlatVector<int64_t>({1, 6}),
       makeFlatVector<int64_t>({3, 3})});

  auto results = AssertQueryBuilder(op).copyResults(pool());
  assertEqualVectors(results, expected);
}