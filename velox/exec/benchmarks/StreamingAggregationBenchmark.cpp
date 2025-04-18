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
#include "folly/Benchmark.h"
#include "folly/init/Init.h"

#include "velox/common/memory/Memory.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::test;

namespace {

struct TestCase {
  // Dataset to be processed by the below plans.
  std::vector<RowVectorPtr> data;
  // Aggregates to be applied.
  std::vector<std::string> aggregates;
  // Streaming Aggregation plan that processes data.
  std::shared_ptr<const core::PlanNode> plan;
};

class StreamingAggregationBenchmark : public VectorTestBase {
 public:
  // Makes benchmark for array_agg.
  // @param name Benchmark name.
  // @param payloadType Type of payload column, like REAL for fload data.
  // @param numPayloads Number of payload columns, assume all payload columns
  // have the same type.
  // @param numRows Total number of rows of the dataset.
  // @param numVectors Number of vectors of the dataset.
  void makeArrayAggBenchmark(
      std::string name,
      const TypePtr payloadType,
      size_t numPayloads,
      size_t numRows,
      int64_t numVectors) {
    auto test = std::make_unique<TestCase>();
    test->data = makeData(payloadType, numPayloads, numRows, numVectors);
    test->aggregates = makeArrayAggAggregates(numPayloads);
    test->plan = makeStreamingAggregationPlan(test->data, test->aggregates);

    folly::addBenchmark(__FILE__, name, [plan = &test->plan, this]() {
      run(*plan);
      return 1;
    });

    cases_.push_back(std::move(test));
  }

 private:
  std::vector<RowVectorPtr> makeData(
      const TypePtr payloadType,
      size_t numPayloads,
      size_t numRows,
      int64_t numVectors) {
    std::vector<RowVectorPtr> data;
    switch (payloadType->kind()) {
      case TypeKind::REAL:
        data = makeData<float>(numPayloads, numRows, numVectors);
        break;
      default:
        VELOX_UNREACHABLE(
            "Unsupported payload type {} ", payloadType->kindName());
    }

    assert(!data.empty());
    return data;
  }

  template <typename PayloadType>
  std::vector<RowVectorPtr>
  makeData(size_t numPayloads, size_t numRows, size_t numVectors) {
    std::vector<std::string> names;
    // One grouping key.
    names.reserve(1 + numPayloads);
    names.push_back("k0");
    for (size_t i = 0; i < numPayloads; ++i) {
      names.push_back(fmt::format("c{}", i));
    }

    std::vector<RowVectorPtr> data;
    data.reserve(numVectors);

    vector_size_t totalSize = 0;
    vector_size_t rowsPerVector = numRows / numVectors;
    for (size_t i = 0; i < numVectors; ++i) {
      std::vector<VectorPtr> children;
      auto keys = makeFlatVector<std::string>(
          rowsPerVector, [&](auto row) { return std::to_string(i); });
      children.push_back(keys);

      for (size_t j = 0; j < numPayloads; ++j) {
        auto payload = makeFlatVector<PayloadType>(
            rowsPerVector, [&](auto row) { return totalSize + row + j; });
        children.push_back(payload);
      }

      data.push_back(makeRowVector(names, children));
      totalSize += rowsPerVector;
    }

    return data;
  }

  std::vector<std::string> makeArrayAggAggregates(size_t numPayloads) {
    std::vector<std::string> aggregates;
    for (size_t i = 0; i < numPayloads; ++i) {
      aggregates.push_back(fmt::format("array_agg(c{})", i));
    }
    return aggregates;
  }

  int64_t run(core::PlanNodePtr plan) {
    auto start = getCurrentTimeMicro();
    auto result = exec::test::AssertQueryBuilder(plan).copyResults(pool_.get());
    auto elapsedMicros = getCurrentTimeMicro() - start;
    return elapsedMicros;
  }

  core::PlanNodePtr makeStreamingAggregationPlan(
      const std::vector<RowVectorPtr>& data,
      const std::vector<std::string>& aggregates) {
    return exec::test::PlanBuilder()
        .values(data)
        .streamingAggregation(
            {"k0"}, aggregates, {}, core::AggregationNode::Step::kSingle, false)
        .planNode();
  }

  std::vector<std::unique_ptr<TestCase>> cases_;
};

} // namespace

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  memory::initializeMemoryManager({});
  aggregate::prestosql::registerAllAggregateFunctions();

  StreamingAggregationBenchmark bm;
  bm.makeArrayAggBenchmark(
      "Float_baseline_repeatKeyOnce", REAL(), 1000, 150, 150);
  bm.makeArrayAggBenchmark("Float_repeatKey5Times", REAL(), 1000, 150, 30);
  bm.makeArrayAggBenchmark(
      "Float_baseline_repeatKey10Times", REAL(), 1000, 150, 15);
  bm.makeArrayAggBenchmark(
      "Float_baseline_repeatKey15Times", REAL(), 1000, 150, 10);
  bm.makeArrayAggBenchmark(
      "Float_baseline_repeatKey30Times", REAL(), 1000, 150, 5);

  folly::runBenchmarks();
  return 0;
}
