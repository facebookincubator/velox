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

#include "velox/core/QueryConfig.h"
#include "velox/dwio/common/tests/utils/BatchMaker.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

/// Benchmark for MergingVectorOutput by different vector size. Merge the
/// output vector of filter can improve the downstream operator's performance,
/// But merge the vector also have additional costs, in this benchmark, we
/// choose FilterProject with merging output, and aggregation as downstream
/// operator, to determine the optimization and the cost of merging,
///
/// Benchmarks run tow plan:
///   1 Filter and aggregation,
///      * The input data are two 1000 * 10K row vectors.
///      * The filter pass percents contain 0.02%, 0.16%, 0.32%, 1%, 10%
///      * Group keys contain 1, 2, 4 group keys.
///      * Data types contain bigint, varchar and complex type array.
///      * Aggregation functions contain count, sum, max.
///
/// String data is benchmarked with either flat or dictionary encoded
/// input. The dictionary encoded case is either with a different set
/// of base values in each vector or each vector sharing the same base
/// values.

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::test;

namespace {
struct TestCase {
  // Dataset to be processed by the below plans.
  std::vector<RowVectorPtr> rows;

  // Dataset for join build side
  std::vector<RowVectorPtr> joinBuildRows;

  // Plan with filter pass 10000 * 10% = 1000
  core::PlanNodePtr plan1000;
  // Plan with filter pass 10000 * 1% = 100
  core::PlanNodePtr plan100;
  // Plan with filter pass 10000 * 0.32% = 32
  core::PlanNodePtr plan32;
  // Plan with filter pass 10000 * 0.16% = 16
  core::PlanNodePtr plan16;
  // Plan with filter pass 10000 * 0.02% = 2
  core::PlanNodePtr plan2;
};

class MergingVectorOutputBenchmark : public VectorTestBase {
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

  template <typename T = int64_t>
  void
  setRandomArrays(int32_t column, int32_t max, std::vector<RowVectorPtr> rows) {
    for (auto& r : rows) {
      if (VectorEncoding::Simple::ARRAY == r->childAt(column)->encoding()) {
        auto arrayVector = r->childAt(column)->as<ArrayVector>();
        auto& arrayType = arrayVector->type()->as<TypeKind::ARRAY>();
        switch (arrayType.childAt(0)->kind()) {
          case TypeKind::BIGINT:
            auto arrays = arrayVector->elements()->as<FlatVector<T>>();
            auto sizeSize = arrays->size();
            // arrays->size() is random size.(49471)
            for (auto i = 0; i < sizeSize; ++i) {
              arrays->set(i, folly::Random::rand32(rng_) % max);
            }
            break;
        }
      }
    }
  }

  core::PlanNodePtr makeFilterAndAggregationPlan(
      float passPct,
      std::shared_ptr<TestCase> testCase) {
    assert(!testCase->rows.empty());
    exec::test::PlanBuilder builder;
    auto& type = testCase->rows[0]->type()->as<TypeKind::ROW>();
    builder.values(testCase->rows);
    builder.filter(fmt::format(
        "k0 >= {}",
        static_cast<int32_t>(maxRandomInt - (passPct / 100.0) * maxRandomInt)));

    std::vector<std::string> projections = {};
    for (auto i = 0; i < type.size(); ++i) {
      projections.push_back(type.nameOf(i));
    }
    builder.project(projections);

    std::vector<std::string> aggregates = {"count(1)"};
    std::vector<std::string> finalProjection;
    bool needFinalProjection = false;
    for (auto i = 0; i < type.size(); ++i) {
      auto columnName = type.nameOf(i);
      finalProjection.push_back(columnName);
      // group by key, should not aggregate
      if (columnName[0] == 'k') {
        continue ;
      }
      switch (type.childAt(i)->kind()) {
        case TypeKind::BIGINT:
          aggregates.push_back(fmt::format("avg({})", columnName));
          break;
        case TypeKind::VARCHAR:
          needFinalProjection = true;
          finalProjection.back() = fmt::format("length({}) as {}", columnName, columnName);
          aggregates.push_back(fmt::format("max({})", columnName));
          break;
        default:
          break;
      }
    }
    if (needFinalProjection) {
      builder.project(finalProjection);
    }
    std::vector<std::string> groupingKeys;
    for (auto i = 0; i < type.size(); ++i) {
      auto columnName = type.nameOf(i);
      if (columnName[0] == 'k') {
        groupingKeys.push_back(columnName);
      }
    }
    builder.singleAggregation(groupingKeys, aggregates);
    return builder.planNode();
  }

  core::PlanNodePtr makeFilterAndJoinPlan(
      float passPct,
      std::shared_ptr<TestCase> testCase) {
    assert(!testCase->rows.empty());
    assert(!testCase->joinBuildRows.empty());
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    int32_t max =
        static_cast<int32_t>(maxRandomInt - (passPct / 100.0) * maxRandomInt);
    exec::test::PlanBuilder builder(planNodeIdGenerator, pool_.get());
    auto& type = testCase->rows[0]->type()->as<TypeKind::ROW>();
    builder.values(testCase->rows);
    builder.filter(fmt::format("k0 >= {}", max));

    std::vector<std::string> projections = {"k0"};
    for (auto i = 1; i < type.size(); ++i) {
      projections.push_back(fmt::format("c{}", i));
    }
    builder.project(projections);

    exec::test::PlanBuilder joinBuildPlanBuilder(
        planNodeIdGenerator, pool_.get());
    joinBuildPlanBuilder.values(testCase->joinBuildRows);
    joinBuildPlanBuilder.filter(fmt::format("r0 >= {}", max));
    joinBuildPlanBuilder.project({"r0"});

    std::vector<std::string> finalProjections = {"r0"};
    for (auto i = 0; i < type.size(); ++i) {
      projections.push_back(fmt::format("c{}", i));
    }
    builder.hashJoin(
        {"k0"}, {"r0"}, joinBuildPlanBuilder.planNode(), "", finalProjections);
    return builder.planNode();
  }

  core::PlanNodePtr makePlan(
      bool isFilterAggregation,
      float passPct,
      std::shared_ptr<TestCase> testCase) {
    if (isFilterAggregation) {
      return makeFilterAndAggregationPlan(passPct, testCase);
    } else {
      return makeFilterAndJoinPlan(passPct, testCase);
    }
  }

  std::string makeString(int32_t n) {
    static std::vector<std::string> tokens = {
        "epi",         "plectic",  "cary",    "ally",    "ously",
        "sly",         "suspect",  "account", "apo",     "thetic",
        "hypo",        "hyper",    "nice",    "fluffy",  "hippocampus",
        "comfortable", "cucurbit", "lemon",   "avocado", "specious",
        "phrenic"};
    std::string result;
    while (n > 0) {
      result = result + tokens[n % tokens.size()];
      n /= tokens.size();
    }
    return result;
  }

  VectorPtr randomStrings(int32_t size, int32_t cardinality) {
    std::string temp;
    return makeFlatVector<StringView>(size, [&](auto /*row*/) {
      temp = makeString(folly::Random::rand32() % cardinality);
      return StringView(temp);
    });
  }

  void prepareStringColumns(
      std::vector<RowVectorPtr> rows,
      int32_t cardinality,
      bool dictionaryStrings,
      bool shareStringDicts,
      bool stringNulls) {
    assert(!rows.empty());
    auto type = rows[0]->type()->as<TypeKind::ROW>();
    auto numColumns = rows[0]->type()->size();
    for (auto column = 0; column < numColumns; ++column) {
      if (type.childAt(column)->kind() == TypeKind::VARCHAR) {
        VectorPtr strings;
        if (dictionaryStrings && shareStringDicts) {
          strings = randomStrings(cardinality, cardinality * 2);
        }
        for (auto row : rows) {
          VectorPtr values;
          if (dictionaryStrings) {
            if (!shareStringDicts) {
              strings = randomStrings(cardinality, cardinality * 2);
            }
            auto indices = makeIndices(row->size(), [&](auto /*row*/) {
              return folly::Random::rand32() % strings->size();
            });
            values = BaseVector::wrapInDictionary(
                nullptr, indices, row->size(), strings);
          } else {
            values = randomStrings(row->size(), cardinality);
          }
          if (stringNulls) {
            setNulls(values, [&](auto row) { return row % 11 == 0; });
          }
          row->childAt(column) = values;
        }
      }
    }
  }

  void addBenchmark(
      std::string name,
      std::shared_ptr<TestCase> test,
      bool isFilterAndAggregation) {
    test->plan1000 = makePlan(isFilterAndAggregation, 10, test);
    test->plan100 = makePlan(isFilterAndAggregation, 1, test);
    test->plan32 = makePlan(isFilterAndAggregation, 0.32, test);
    test->plan16 = makePlan(isFilterAndAggregation, 0.16, test);
    test->plan2 = makePlan(isFilterAndAggregation, 0.02, test);

    folly::addBenchmark(
        __FILE__, name + "2_mergeOff", [plan = &test->plan2, this]() {
          run(*plan, 0);
          return 1;
        });
    folly::addBenchmark(
        __FILE__, name + "2_mergeOn", [plan = &test->plan2, this]() {
          run(*plan, 16);
          return 1;
        });

    folly::addBenchmark(
        __FILE__, name + "16_mergeOff", [plan = &test->plan16, this]() {
          run(*plan, 0);
          return 1;
        });
    folly::addBenchmark(
        __FILE__, name + "16_mergeOn", [plan = &test->plan16, this]() {
          run(*plan, 16 * 2);
          return 1;
        });

    folly::addBenchmark(
        __FILE__, name + "32_mergeOff", [plan = &test->plan32, this]() {
          run(*plan, 0);
          return 1;
        });
    folly::addBenchmark(
        __FILE__, name + "32_mergeOn", [plan = &test->plan32, this]() {
          run(*plan, 32 * 2);
          return 1;
        });

    folly::addBenchmark(
        __FILE__, name + "100_mergeOff", [plan = &test->plan100, this]() {
          run(*plan, 0);
          return 1;
        });
    folly::addBenchmark(
        __FILE__, name + "100_mergeOn", [plan = &test->plan100, this]() {
          run(*plan, 100 * 2);
          return 1;
        });

    folly::addBenchmark(
        __FILE__, name + "1000_mergeOff", [plan = &test->plan1000, this]() {
          run(*plan, 0);
          return 1;
        });
    folly::addBenchmark(
        __FILE__, name + "1000_mergeOn", [plan = &test->plan1000, this]() {
          run(*plan, 1000 * 2);
          return 1;
        });

    cases_.push_back(std::move(test));
  }

  void makeBenchmark(
      std::string name,
      RowTypePtr type,
      int64_t numVectors,
      int32_t numPerVector,
      int32_t stringCardinality = 1000,
      bool dictionaryStrings = false,
      bool shareStringDicts = false,
      bool stringNulls = false) {
    auto test = std::make_shared<TestCase>();
    test->rows = makeRows(type, numVectors, numPerVector);

    for (auto i = 0; i < type->size(); ++i) {
      switch (type->childAt(i)->kind()) {
        case TypeKind::BIGINT:
          setRandomInts(i, maxRandomInt, test->rows);
          break;
        case TypeKind::ARRAY:
          setRandomArrays(i, maxRandomInt, test->rows);
          break;
        default:
          break;
      }
    }

    prepareStringColumns(
        test->rows,
        stringCardinality,
        dictionaryStrings,
        shareStringDicts,
        stringNulls);
    test->joinBuildRows =
        makeRows(ROW({{"r0", BIGINT()}}), numVectors, numPerVector);
    setRandomInts(0, maxRandomInt, test->joinBuildRows);

    addBenchmark(name + "filter_agg_pass", test, true);
    //addBenchmark(name + "filter_join_pass", test, false);
  }

  int64_t run(core::PlanNodePtr plan, int32_t minRows) {
    auto start = getCurrentTimeMicro();
    int32_t numRows = 0;
    auto result = exec::test::AssertQueryBuilder(plan)
                      .config(
                          facebook::velox::core::QueryConfig::
                              kMinMergingVectorOutputBatchRows,
                          std::to_string(minRows))
                      .copyResults(pool_.get());
    numRows += result->childAt(0)->as<FlatVector<int64_t>>()->valueAt(0);
    auto elapsedMicros = getCurrentTimeMicro() - start;
    return elapsedMicros;
  }

  std::vector<std::shared_ptr<TestCase>> cases_;
  folly::Random::DefaultGenerator rng_;
  int32_t maxRandomInt = 100'000'000;
};
} // namespace

int main(int argc, char** argv) {
  folly::init(&argc, &argv);
  functions::prestosql::registerAllScalarFunctions();
  aggregate::prestosql::registerAllAggregateFunctions();
  parse::registerTypeResolver();

  MergingVectorOutputBenchmark bm;

  auto arrayType = ARRAY(BIGINT());

  auto simpleType = ROW(
      {{"k0", BIGINT()},
       {"c1", BIGINT()},
       {"c2", BIGINT()},
       {"c3", VARCHAR()}});

  auto group2Keys = ROW(
      {{"k0", BIGINT()},
       {"k1", BIGINT()},
       {"c2", BIGINT()},
       {"c3", VARCHAR()}});

  auto group4Keys = ROW(
      {{"k0", BIGINT()},
       {"k1", BIGINT()},
       {"k2", BIGINT()},
       {"k3", BIGINT()},
       {"c5", BIGINT()}});

  auto complexType = ROW(
      {{"k0", BIGINT()},
       {"c1", BIGINT()},
       {"c2", BIGINT()},
       {"c3", arrayType}});

  int32_t numVectors = 1000;
  int32_t rowsPerVector = 10000;

  // Flat simple types.
  bm.makeBenchmark("SimpleType_", simpleType, numVectors, rowsPerVector);

  bm.makeBenchmark("Group2Keys_", group2Keys, numVectors, rowsPerVector);

  bm.makeBenchmark("Group4Keys_", group4Keys, numVectors, rowsPerVector);

  // Flat complexType.
  bm.makeBenchmark("ComplexType_", complexType, numVectors, rowsPerVector);

  // Strings dictionary encoded.
  bm.makeBenchmark(
      "StringDict_",
      simpleType,
      numVectors,
      rowsPerVector,
      200,
      true,
      false,
      true);

  // Strings with dictionary base values shared between batches.
  bm.makeBenchmark(
      "StringRepDict_",
      simpleType,
      numVectors,
      rowsPerVector,
      200,
      true,
      true,
      true);

  folly::runBenchmarks();
  return 0;
}
