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

#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/aggregates/tests/AggregationTestBase.h"
#include "velox/functions/sparksql/aggregates/Register.h"

namespace facebook::velox::functions::sparksql::aggregates::test {
namespace {
class BloomFilterAggAggregateTest
    : public aggregate::test::AggregationTestBase {
 public:
  BloomFilterAggAggregateTest() {
    aggregate::test::AggregationTestBase::SetUp();
    aggregates::registerAggregateFunctions("");
  }
};
} // namespace

TEST_F(BloomFilterAggAggregateTest, bloomFilter) {
  auto vectors = {makeRowVector({makeFlatVector<int64_t>(
      100, [](vector_size_t row) { return row / 3; })})};

  auto expected = {makeRowVector({makeFlatVector<
      StringView>(1, [](vector_size_t row) {
    return "\u0004\u0000\u0000\u0000\u0003\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000";
  })})};

  testAggregations(vectors, {}, {"bloom_filter_agg(c0, 5, 10)"}, expected);
}

} // namespace facebook::velox::functions::sparksql::aggregates::test
