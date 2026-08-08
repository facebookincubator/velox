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

#include "velox/exec/Cursor.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/lib/aggregates/ValueList.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

DEFINE_int64(fuzzer_seed, 99887766, "Seed for random input dataset generator");

using namespace facebook::velox;
using namespace facebook::velox::aggregate;
using namespace facebook::velox::exec::test;

namespace {

// End-to-end benchmark: SELECT array_agg(x) FROM t (global, single group)
// exercises Aggregate::addSingleGroupRawInput, which is the path the
// ValueList::appendValues optimization targets.
class ArrayAggBenchmark
    : public facebook::velox::exec::test::HiveConnectorTestBase {
 public:
  ArrayAggBenchmark() {
    HiveConnectorTestBase::SetUp();

    inputType_ = ROW({
        {"i32", INTEGER()},
        {"i64", BIGINT()},
        {"f64", DOUBLE()},
        {"i64_halfnull", BIGINT()},
        {"str_short", VARCHAR()},
        {"str_long", VARCHAR()},
        {"str_halfnull", VARCHAR()},
    });

    VectorFuzzer::Options opts;
    opts.vectorSize = kRowsPerVector;
    opts.nullRatio = 0;
    VectorFuzzer fuzzer(opts, pool(), FLAGS_fuzzer_seed);

    std::vector<RowVectorPtr> vectors;
    for (auto i = 0; i < kNumVectors; ++i) {
      std::vector<VectorPtr> children;
      children.emplace_back(fuzzer.fuzzFlat(INTEGER()));
      children.emplace_back(fuzzer.fuzzFlat(BIGINT()));
      children.emplace_back(fuzzer.fuzzFlat(DOUBLE()));
      opts.nullRatio = 0.5;
      fuzzer.setOptions(opts);
      children.emplace_back(fuzzer.fuzzFlat(BIGINT()));
      opts.nullRatio = 0;
      fuzzer.setOptions(opts);
      opts.stringLength = 10;
      fuzzer.setOptions(opts);
      children.emplace_back(fuzzer.fuzzFlat(VARCHAR()));
      opts.stringLength = 200;
      fuzzer.setOptions(opts);
      children.emplace_back(fuzzer.fuzzFlat(VARCHAR()));
      opts.nullRatio = 0.5;
      opts.stringLength = 50;
      fuzzer.setOptions(opts);
      children.emplace_back(fuzzer.fuzzFlat(VARCHAR()));
      opts.nullRatio = 0;
      opts.stringLength = 0;
      fuzzer.setOptions(opts);
      vectors.emplace_back(makeRowVector(inputType_->names(), children));
    }

    filePath_ = TempFilePath::create();
    writeToFile(filePath_->getPath(), vectors);
  }

  ~ArrayAggBenchmark() override {
    HiveConnectorTestBase::TearDown();
  }

  void TestBody() override {}

  // Global (single-group) array_agg. Exercises addSingleGroupRawInput.
  void runGlobal(const std::string& aggregate) {
    folly::BenchmarkSuspender suspender;
    auto plan = PlanBuilder()
                    .tableScan(inputType_)
                    .singleAggregation({}, {aggregate})
                    .planFragment();

    auto task = makeTask(plan);
    task->addSplit(
        "0",
        facebook::velox::exec::Split(
            makeHiveConnectorSplit(filePath_->getPath())));
    task->noMoreSplits("0");
    suspender.dismiss();

    vector_size_t numResultRows = 0;
    while (auto result = task->next()) {
      numResultRows += result->size();
    }
    folly::doNotOptimizeAway(numResultRows);
  }

  std::shared_ptr<facebook::velox::exec::Task> makeTask(
      core::PlanFragment plan) {
    return facebook::velox::exec::Task::create(
        "t",
        std::move(plan),
        0,
        core::QueryCtx::create(executor_.get()),
        facebook::velox::exec::Task::ExecutionMode::kSerial,
        facebook::velox::exec::Consumer{});
  }

 public:
  static constexpr int32_t kNumVectors = 200;
  static constexpr int32_t kRowsPerVector = 10'000;

 private:
  RowTypePtr inputType_;
  std::shared_ptr<TempFilePath> filePath_;
};

std::unique_ptr<ArrayAggBenchmark> benchmark;

void doRunGlobal(uint32_t, const std::string& aggregate) {
  benchmark->runGlobal(aggregate);
}

BENCHMARK_NAMED_PARAM(doRunGlobal, global_i32, "array_agg(i32)");
BENCHMARK_NAMED_PARAM(doRunGlobal, global_i64, "array_agg(i64)");
BENCHMARK_NAMED_PARAM(doRunGlobal, global_f64, "array_agg(f64)");
BENCHMARK_NAMED_PARAM(
    doRunGlobal,
    global_i64_halfnull,
    "array_agg(i64_halfnull)");
BENCHMARK_NAMED_PARAM(doRunGlobal, global_str_short, "array_agg(str_short)");
BENCHMARK_NAMED_PARAM(doRunGlobal, global_str_long, "array_agg(str_long)");
BENCHMARK_NAMED_PARAM(
    doRunGlobal,
    global_str_halfnull,
    "array_agg(str_halfnull)");
BENCHMARK_DRAW_LINE();

// Direct ValueList microbenchmark: compares appendValue (per-row stream
// extendWrite/finishWrite) against appendValues (single batched session).
class ValueListMicroBenchmark : public test::VectorTestBase {
 public:
  ValueListMicroBenchmark() {
    constexpr int32_t kSize = 100'000;
    inputInt_ = makeFlatVector<int64_t>(kSize, [](auto i) { return i * 31; });

    VectorFuzzer::Options opts;
    opts.vectorSize = kSize;
    opts.nullRatio = 0;
    opts.stringLength = 10;
    VectorFuzzer fuzzer(opts, pool(), 42);
    inputStrShort_ = fuzzer.fuzzFlat(VARCHAR());
    opts.stringLength = 200;
    fuzzer.setOptions(opts);
    inputStr200_ = fuzzer.fuzzFlat(VARCHAR());
    opts.stringLength = 1'000;
    fuzzer.setOptions(opts);
    inputStr1K_ = fuzzer.fuzzFlat(VARCHAR());
    opts.stringLength = 10'000;
    fuzzer.setOptions(opts);
    inputStr10K_ = fuzzer.fuzzFlat(VARCHAR());
    opts.vectorSize = 10'000;
    opts.stringLength = 100'000;
    fuzzer.setOptions(opts);
    inputStr100K_ = fuzzer.fuzzFlat(VARCHAR());
    rows10K_ = SelectivityVector(10'000);
    opts.vectorSize = 1'000;
    opts.stringLength = 500'000;
    fuzzer.setOptions(opts);
    inputStr500K_ = fuzzer.fuzzFlat(VARCHAR());
    rows1K_ = SelectivityVector(1'000);

    rows_ = SelectivityVector(kSize);
  }

  void appendOneByOne(const VectorPtr& input, const SelectivityVector& rows) {
    HashStringAllocator allocator{pool()};
    DecodedVector decoded(*input, rows);
    ValueList values;
    rows.applyToSelected([&](vector_size_t row) {
      values.appendValue(decoded, row, &allocator);
    });
    folly::doNotOptimizeAway(values.size());
    values.free(&allocator);
  }

  void appendBatch(const VectorPtr& input, const SelectivityVector& rows) {
    HashStringAllocator allocator{pool()};
    DecodedVector decoded(*input, rows);
    ValueList values;
    values.appendValues(decoded, rows, &allocator);
    folly::doNotOptimizeAway(values.size());
    values.free(&allocator);
  }

  VectorPtr inputInt_;
  VectorPtr inputStrShort_;
  VectorPtr inputStr200_;
  VectorPtr inputStr1K_;
  VectorPtr inputStr10K_;
  VectorPtr inputStr100K_;
  VectorPtr inputStr500K_;
  SelectivityVector rows_;
  SelectivityVector rows10K_;
  SelectivityVector rows1K_;
};

std::unique_ptr<ValueListMicroBenchmark> micro;

BENCHMARK(valueList_int64_perRow) {
  micro->appendOneByOne(micro->inputInt_, micro->rows_);
}

BENCHMARK_RELATIVE(valueList_int64_batch) {
  micro->appendBatch(micro->inputInt_, micro->rows_);
}

BENCHMARK_DRAW_LINE();

BENCHMARK(valueList_str10_perRow) {
  micro->appendOneByOne(micro->inputStrShort_, micro->rows_);
}

BENCHMARK_RELATIVE(valueList_str10_batch) {
  micro->appendBatch(micro->inputStrShort_, micro->rows_);
}

BENCHMARK_DRAW_LINE();

BENCHMARK(valueList_str200_perRow) {
  micro->appendOneByOne(micro->inputStr200_, micro->rows_);
}

BENCHMARK_RELATIVE(valueList_str200_batch) {
  micro->appendBatch(micro->inputStr200_, micro->rows_);
}

BENCHMARK_DRAW_LINE();

BENCHMARK(valueList_str1K_perRow) {
  micro->appendOneByOne(micro->inputStr1K_, micro->rows_);
}

BENCHMARK_RELATIVE(valueList_str1K_batch) {
  micro->appendBatch(micro->inputStr1K_, micro->rows_);
}

BENCHMARK_DRAW_LINE();

BENCHMARK(valueList_str10K_perRow) {
  micro->appendOneByOne(micro->inputStr10K_, micro->rows_);
}

BENCHMARK_RELATIVE(valueList_str10K_batch) {
  micro->appendBatch(micro->inputStr10K_, micro->rows_);
}

BENCHMARK_DRAW_LINE();

BENCHMARK(valueList_str100K_perRow) {
  micro->appendOneByOne(micro->inputStr100K_, micro->rows10K_);
}

BENCHMARK_RELATIVE(valueList_str100K_batch) {
  micro->appendBatch(micro->inputStr100K_, micro->rows10K_);
}

BENCHMARK_DRAW_LINE();

BENCHMARK(valueList_str500K_perRow) {
  micro->appendOneByOne(micro->inputStr500K_, micro->rows1K_);
}

BENCHMARK_RELATIVE(valueList_str500K_batch) {
  micro->appendBatch(micro->inputStr500K_, micro->rows1K_);
}

} // namespace

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  facebook::velox::exec::test::OperatorTestBase::SetUpTestCase();
  benchmark = std::make_unique<ArrayAggBenchmark>();
  micro = std::make_unique<ValueListMicroBenchmark>();
  folly::runBenchmarks();
  benchmark.reset();
  micro.reset();
  facebook::velox::exec::test::OperatorTestBase::TearDownTestCase();
  return 0;
}
