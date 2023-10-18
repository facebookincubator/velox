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
#include <folly/Benchmark.h>
#include <folly/init/Init.h>

#include "velox/dwio/common/tests/utils/BatchMaker.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::test;

namespace {
struct TestCase {
  // Dataset to be processed by the below plans.
  std::vector<RowVectorPtr> rows;

  std::shared_ptr<const core::PlanNode> _1key;
  std::shared_ptr<const core::PlanNode> _2key;
  std::shared_ptr<const core::PlanNode> _3key;
};

class PrefixSortBenchmark : public VectorTestBase {
 public:
  std::vector<RowVectorPtr>
  makeRows(RowTypePtr type, int32_t numVectors, int32_t rowsPerVector) {
    std::vector<RowVectorPtr> vectors;
    for (int32_t i = 0; i < numVectors; ++i) {
      auto vector = std::dynamic_pointer_cast<RowVector>(
          BatchMaker::createBatch(type, rowsPerVector, *pool_));
      vectors.push_back(vector);
    }
    return vectors;
  }

  template <typename T = int64_t>
  void
  setRandomInts(int32_t column, int32_t max, std::vector<RowVectorPtr> rows) {
    for (auto& r : rows) {
      auto values = r->childAt(column)->as<FlatVector<T>>();
      for (auto i = 0; i < values->size(); ++i) {
        values->set(i, folly::Random::rand32(rng_) % max);
      }
    }
  }

  std::shared_ptr<const core::PlanNode> makeOrderByPlan(
      std::vector<std::string> keys,
      std::vector<RowVectorPtr> data) {
    assert(!data.empty());
    exec::test::PlanBuilder builder;

    auto& type = data[0]->type()->as<TypeKind::ROW>();
    builder.values(data);
    builder.orderBy(keys, false);

    return builder.planNode();
  }

  void makeBenchmark(
      std::string name,
      RowTypePtr type,
      int64_t numVectors,
      int32_t numPerVector,
      int32_t stringCardinality = 1000) {
    auto test = std::make_unique<TestCase>();
    test->rows = makeRows(type, numVectors, numPerVector);
    // low selectivity for full compare
    setRandomInts(0, 1, test->rows);
    setRandomInts(1, 1, test->rows);
    setRandomInts(2, 10000000, test->rows);

    test->_1key = makeOrderByPlan({"c2"}, test->rows);
    folly::addBenchmark(
        __FILE__, name + "_1key_base", [plan = &test->_1key, this]() {
          run(*plan, "false");
          return 1;
        });
    folly::addBenchmark(
        __FILE__, name + "_1key_prefix_sort", [plan = &test->_1key, this]() {
          run(*plan, "true");
          return 1;
        });
    test->_2key = makeOrderByPlan({"c1", "c2"}, test->rows);
    folly::addBenchmark(
        __FILE__, name + "_2key_base", [plan = &test->_2key, this]() {
          run(*plan, "false");
          return 1;
        });
    folly::addBenchmark(
        __FILE__, name + "_2key_prefix_sort", [plan = &test->_2key, this]() {
          run(*plan, "true");
          return 1;
        });
    test->_3key = makeOrderByPlan({"c0", "c1", "c2"}, test->rows);
    folly::addBenchmark(
        __FILE__, name + "_3key_base", [plan = &test->_3key, this]() {
          run(*plan, "false");
          return 1;
        });
    folly::addBenchmark(
        __FILE__, name + "_3key_prefix_sort", [plan = &test->_3key, this]() {
          run(*plan, "true");
          return 1;
        });

    cases_.push_back(std::move(test));
  }

  int64_t run(
      std::shared_ptr<const core::PlanNode> plan,
      const std::string& enablePrefixSort) {
    auto start = getCurrentTimeMicro();
    int32_t numRows = 0;
    auto result = exec::test::AssertQueryBuilder(plan)
                      .config(
                          facebook::velox::core::QueryConfig::kEnablePrefixSort,
                          enablePrefixSort)
                      .copyResults(pool_.get());
    numRows += result->childAt(0)->as<FlatVector<int64_t>>()->valueAt(0);
    auto elapsedMicros = getCurrentTimeMicro() - start;
    return elapsedMicros;
  }

  std::vector<std::unique_ptr<TestCase>> cases_;
  folly::Random::DefaultGenerator rng_;
};
} // namespace

int main(int argc, char** argv) {
  folly::init(&argc, &argv);
  functions::prestosql::registerAllScalarFunctions();
  aggregate::prestosql::registerAllAggregateFunctions();
  parse::registerTypeResolver();

  PrefixSortBenchmark bm;

  auto bigint3 = ROW(
      {{"c0", BIGINT()}, {"c1", BIGINT()}, {"c2", BIGINT()}, {"c3", BIGINT()}});

  // Integers.
  bm.makeBenchmark("Bigint_100K", bigint3, 10, 10000);
  bm.makeBenchmark("Bigint_1000K", bigint3, 100, 10000);
  bm.makeBenchmark("Bigint_10000K", bigint3, 1000, 10000);

  folly::runBenchmarks();
  return 0;
}
