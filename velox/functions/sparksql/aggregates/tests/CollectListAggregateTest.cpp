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

#include "velox/exec/tests/SimpleAggregateFunctionsRegistration.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/lib/aggregates/tests/utils/AggregationTestBase.h"
#include "velox/functions/sparksql/aggregates/Register.h"

using namespace facebook::velox::functions::aggregate::test;
using facebook::velox::exec::test::AssertQueryBuilder;
using facebook::velox::exec::test::PlanBuilder;

namespace facebook::velox::functions::aggregate::sparksql::test {

namespace {

class CollectListAggregateTest : public AggregationTestBase {
 protected:
  void SetUp() override {
    AggregationTestBase::SetUp();
    registerAggregateFunctions("spark_");
  }

  RowVectorPtr fuzzFlat(const RowTypePtr& rowType, size_t size) {
    VectorFuzzer::Options options;
    options.vectorSize = size;
    VectorFuzzer fuzzer(options, pool());
    return fuzzer.fuzzInputFlatRow(rowType);
  }
};

TEST_F(CollectListAggregateTest, groupBy) {
  constexpr int32_t kNumGroups = 10;
  std::vector<RowVectorPtr> batches;
  batches.push_back(
      fuzzFlat(ROW({"c0", "a"}, {INTEGER(), ARRAY(VARCHAR())}), 100));
  auto keys = batches[0]->childAt(0)->as<FlatVector<int32_t>>();
  auto values = batches[0]->childAt(1)->as<ArrayVector>();
  for (auto i = 0; i < keys->size(); ++i) {
    if (i % 10 == 0) {
      keys->setNull(i, true);
    } else {
      keys->set(i, i % kNumGroups);
    }

    if (i % 7 == 0) {
      values->setNull(i, true);
    }
  }

  for (auto i = 0; i < 9; ++i) {
    batches.push_back(batches[0]);
  }

  createDuckDbTable(batches);
  testAggregations(
      batches,
      {"c0"},
      {"spark_collect_list(a)"},
      {"c0", "array_sort(a0)"},
      "SELECT c0, array_sort(array_agg(a)"
      "filter (where a is not null)) FROM tmp GROUP BY c0");
  testAggregationsWithCompanion(
      batches,
      [](auto& /*builder*/) {},
      {"c0"},
      {"spark_collect_list(a)"},
      {{ARRAY(VARCHAR())}},
      {"c0", "array_sort(a0)"},
      "SELECT c0, array_sort(array_agg(a)"
      "filter (where a is not null)) FROM tmp GROUP BY c0");
}

TEST_F(CollectListAggregateTest, global) {
  vector_size_t size = 10;
  std::vector<RowVectorPtr> vectors = {makeRowVector({makeFlatVector<int32_t>(
      size, [](vector_size_t row) { return row * 2; }, nullEvery(3))})};

  createDuckDbTable(vectors);
  testAggregations(
      vectors,
      {},
      {"spark_collect_list(c0)"},
      {"array_sort(a0)"},
      "SELECT array_sort(array_agg(c0)"
      "filter (where c0 is not null)) FROM tmp");
  testAggregationsWithCompanion(
      vectors,
      [](auto& /*builder*/) {},
      {},
      {"spark_collect_list(c0)"},
      {{ARRAY(VARCHAR())}},
      {"array_sort(a0)"},
      "SELECT array_sort(array_agg(c0)"
      "filter (where c0 is not null)) FROM tmp");
}

TEST_F(CollectListAggregateTest, ignoreNulls) {
  auto input = makeRowVector({makeNullableFlatVector<int32_t>(
      {1, 2, std::nullopt, 4, std::nullopt, 6}, INTEGER())});
  // Spark will ignore all null values in the input.
  auto expected = makeRowVector({makeArrayVector<int32_t>({{1, 2, 4, 6}})});
  testAggregations(
      {input}, {}, {"spark_collect_list(c0)"}, {"array_sort(a0)"}, {expected});
  testAggregationsWithCompanion(
      {input},
      [](auto& /*builder*/) {},
      {},
      {"spark_collect_list(c0)"},
      {{INTEGER()}},
      {"array_sort(a0)"},
      {expected},
      {});
}

TEST_F(CollectListAggregateTest, allNullsInput) {
  std::vector<std::optional<int64_t>> allNull(100, std::nullopt);
  auto input =
      makeRowVector({makeNullableFlatVector<int64_t>(allNull, BIGINT())});
  // If all input data is null, Spark will output an empty array.
  auto expected = makeRowVector({makeArrayVector<int32_t>({{}})});
  testAggregations({input}, {}, {"spark_collect_list(c0)"}, {expected});
  testAggregationsWithCompanion(
      {input},
      [](auto& /*builder*/) {},
      {},
      {"spark_collect_list(c0)"},
      {{BIGINT()}},
      {},
      {expected},
      {});
}
} // namespace
} // namespace facebook::velox::functions::aggregate::sparksql::test
