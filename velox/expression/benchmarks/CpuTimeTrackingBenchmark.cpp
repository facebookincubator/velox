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

/// Microbenchmark to measure the overhead of CPU time tracking
/// (expression.track_cpu_usage) and adaptive per-function sampling
/// (expression.adaptive_cpu_sampling) on simple scalar function evaluation.
///
/// CPU time tracking uses clock_gettime(CLOCK_THREAD_CPUTIME_ID) on every
/// function invocation, which can be expensive relative to the function body
/// for cheap functions like multiply. Adaptive sampling calibrates per-function
/// overhead and applies variable sampling rates to keep overhead bounded.

#include <folly/Benchmark.h>
#include <folly/init/Init.h>
#include <gflags/gflags.h>

#include "velox/functions/Macros.h"
#include "velox/functions/Registerer.h"
#include "velox/functions/lib/benchmarks/FunctionBenchmarkBase.h"
#include "velox/functions/prestosql/ArithmeticImpl.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

DEFINE_int64(
    fuzzer_seed,
    99'887'766,
    "Seed for random input dataset generator");

using namespace facebook::velox;
using namespace facebook::velox::exec;

namespace {

template <typename T>
struct MultiplyFunction {
  template <typename TInput>
  FOLLY_ALWAYS_INLINE void
  call(TInput& result, const TInput& a, const TInput& b) {
    result = functions::multiply(a, b);
  }
};

/// Element-wise >= comparison on two int64 arrays. Mirrors the core logic of
/// array_gte UDF — a representative "expensive" function because it
/// iterates over array elements, allocates an output array, and touches more
/// memory per row than a simple scalar op like multiply.
template <typename T>
struct ArrayGteFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE void callNullFree(
      out_type<Array<bool>>& out,
      const null_free_arg_type<Array<int64_t>>& lhs,
      const null_free_arg_type<Array<int64_t>>& rhs) {
    auto size = std::min(lhs.size(), rhs.size());
    out.reserve(size);
    for (auto i = 0; i < size; ++i) {
      out.push_back(lhs[i] >= rhs[i]);
    }
  }
};

class CpuTimeTrackingBenchmark : public functions::test::FunctionBenchmarkBase {
 public:
  CpuTimeTrackingBenchmark() : FunctionBenchmarkBase() {
    registerFunction<MultiplyFunction, double, double, double>({"multiply"});

    inputType_ = ROW({{"a", DOUBLE()}, {"b", DOUBLE()}});

    smallRowVector_ = makeRowVector(100);
    mediumRowVector_ = makeRowVector(1'000);
    largeRowVector_ = makeRowVector(10'000);

    // Compile with CPU tracking OFF (default).
    exprSetNoTracking_ = compileWithConfig({
        {core::QueryConfig::kExprTrackCpuUsage, "false"},
    });

    // Compile with CPU tracking ON.
    exprSetWithTracking_ = compileWithConfig({
        {core::QueryConfig::kExprTrackCpuUsage, "true"},
    });

    // Compile with adaptive CPU sampling (1% max overhead).
    exprSetAdaptive1Pct_ = compileWithConfig({
        {core::QueryConfig::kExprTrackCpuUsage, "false"},
        {core::QueryConfig::kExprAdaptiveCpuSampling, "true"},
        {core::QueryConfig::kExprAdaptiveCpuSamplingMaxOverheadPct, "1.0"},
    });

    // Compile with adaptive CPU sampling (0.5% max overhead).
    exprSetAdaptiveHalfPct_ = compileWithConfig({
        {core::QueryConfig::kExprTrackCpuUsage, "false"},
        {core::QueryConfig::kExprAdaptiveCpuSampling, "true"},
        {core::QueryConfig::kExprAdaptiveCpuSamplingMaxOverheadPct, "0.5"},
    });
  }

  void runNoTracking(const RowVectorPtr& input, size_t iterations) {
    run(*exprSetNoTracking_, input, iterations);
  }

  void runWithTracking(const RowVectorPtr& input, size_t iterations) {
    run(*exprSetWithTracking_, input, iterations);
  }

  void runAdaptive1Pct(const RowVectorPtr& input, size_t iterations) {
    run(*exprSetAdaptive1Pct_, input, iterations);
  }

  void runAdaptiveHalfPct(const RowVectorPtr& input, size_t iterations) {
    run(*exprSetAdaptiveHalfPct_, input, iterations);
  }

  RowVectorPtr smallRowVector_;
  RowVectorPtr mediumRowVector_;
  RowVectorPtr largeRowVector_;

 private:
  RowVectorPtr makeRowVector(vector_size_t size) {
    VectorFuzzer::Options opts;
    opts.vectorSize = size;
    opts.nullRatio = 0;
    VectorFuzzer fuzzer(opts, pool(), FLAGS_fuzzer_seed);

    std::vector<VectorPtr> children;
    children.emplace_back(fuzzer.fuzzFlat(DOUBLE()));
    children.emplace_back(fuzzer.fuzzFlat(DOUBLE()));
    return std::make_shared<RowVector>(
        pool(), inputType_, nullptr, size, std::move(children));
  }

  std::unique_ptr<ExprSet> compileWithConfig(
      std::unordered_map<std::string, std::string> config) {
    queryCtx_->testingOverrideConfigUnsafe(std::move(config));
    return std::make_unique<ExprSet>(
        compileExpression("multiply(a, b)", inputType_));
  }

  void run(ExprSet& exprSet, const RowVectorPtr& input, size_t iterations) {
    folly::BenchmarkSuspender suspender;
    SelectivityVector rows(input->size());
    suspender.dismiss();

    std::vector<VectorPtr> results(1);
    size_t count{0};
    for (size_t i = 0; i < iterations; ++i) {
      EvalCtx evalCtx(&execCtx_, &exprSet, input.get());
      exprSet.eval(rows, evalCtx, results);
      count += results[0]->size();
      results[0]->prepareForReuse();
    }
    folly::doNotOptimizeAway(count);
  }

  TypePtr inputType_;
  std::unique_ptr<ExprSet> exprSetNoTracking_;
  std::unique_ptr<ExprSet> exprSetWithTracking_;
  std::unique_ptr<ExprSet> exprSetAdaptive1Pct_;
  std::unique_ptr<ExprSet> exprSetAdaptiveHalfPct_;
};

/// Benchmark for array_gte — an expensive UDF that iterates over array
/// elements. Measures CPU tracking and adaptive sampling overhead on a function
/// whose per-row cost is much higher than a simple scalar op.
class ArrayGteBenchmark : public functions::test::FunctionBenchmarkBase {
 public:
  ArrayGteBenchmark() : FunctionBenchmarkBase() {
    registerFunction<
        ArrayGteFunction,
        Array<bool>,
        Array<int64_t>,
        Array<int64_t>>({"array_gte"});

    inputType_ = ROW({{"a", ARRAY(BIGINT())}, {"b", ARRAY(BIGINT())}});

    smallRowVector_ = makeRowVector(100);
    mediumRowVector_ = makeRowVector(1'000);
    largeRowVector_ = makeRowVector(10'000);

    exprSetNoTracking_ = compileWithConfig({
        {core::QueryConfig::kExprTrackCpuUsage, "false"},
    });
    exprSetWithTracking_ = compileWithConfig({
        {core::QueryConfig::kExprTrackCpuUsage, "true"},
    });
    exprSetAdaptive1Pct_ = compileWithConfig({
        {core::QueryConfig::kExprTrackCpuUsage, "false"},
        {core::QueryConfig::kExprAdaptiveCpuSampling, "true"},
        {core::QueryConfig::kExprAdaptiveCpuSamplingMaxOverheadPct, "1.0"},
    });
    exprSetAdaptiveHalfPct_ = compileWithConfig({
        {core::QueryConfig::kExprTrackCpuUsage, "false"},
        {core::QueryConfig::kExprAdaptiveCpuSampling, "true"},
        {core::QueryConfig::kExprAdaptiveCpuSamplingMaxOverheadPct, "0.5"},
    });
  }

  void runNoTracking(const RowVectorPtr& input, size_t iterations) {
    run(*exprSetNoTracking_, input, iterations);
  }

  void runWithTracking(const RowVectorPtr& input, size_t iterations) {
    run(*exprSetWithTracking_, input, iterations);
  }

  void runAdaptive1Pct(const RowVectorPtr& input, size_t iterations) {
    run(*exprSetAdaptive1Pct_, input, iterations);
  }

  void runAdaptiveHalfPct(const RowVectorPtr& input, size_t iterations) {
    run(*exprSetAdaptiveHalfPct_, input, iterations);
  }

  RowVectorPtr smallRowVector_;
  RowVectorPtr mediumRowVector_;
  RowVectorPtr largeRowVector_;

 private:
  RowVectorPtr makeRowVector(vector_size_t size) {
    VectorFuzzer::Options opts;
    opts.vectorSize = size;
    opts.nullRatio = 0;
    opts.containerLength = 50;
    VectorFuzzer fuzzer(opts, pool(), FLAGS_fuzzer_seed);

    std::vector<VectorPtr> children;
    children.emplace_back(fuzzer.fuzzFlat(ARRAY(BIGINT())));
    children.emplace_back(fuzzer.fuzzFlat(ARRAY(BIGINT())));
    return std::make_shared<RowVector>(
        pool(), inputType_, nullptr, size, std::move(children));
  }

  std::unique_ptr<ExprSet> compileWithConfig(
      std::unordered_map<std::string, std::string> config) {
    queryCtx_->testingOverrideConfigUnsafe(std::move(config));
    return std::make_unique<ExprSet>(
        compileExpression("array_gte(a, b)", inputType_));
  }

  void run(ExprSet& exprSet, const RowVectorPtr& input, size_t iterations) {
    folly::BenchmarkSuspender suspender;
    SelectivityVector rows(input->size());
    suspender.dismiss();

    std::vector<VectorPtr> results(1);
    size_t count{0};
    for (size_t i = 0; i < iterations; ++i) {
      EvalCtx evalCtx(&execCtx_, &exprSet, input.get());
      exprSet.eval(rows, evalCtx, results);
      count += results[0]->size();
      results[0]->prepareForReuse();
    }
    folly::doNotOptimizeAway(count);
  }

  TypePtr inputType_;
  std::unique_ptr<ExprSet> exprSetNoTracking_;
  std::unique_ptr<ExprSet> exprSetWithTracking_;
  std::unique_ptr<ExprSet> exprSetAdaptive1Pct_;
  std::unique_ptr<ExprSet> exprSetAdaptiveHalfPct_;
};

constexpr size_t kIterationsSmall{10'000};
constexpr size_t kIterationsMedium{1'000};
constexpr size_t kIterationsLarge{100};

std::unique_ptr<CpuTimeTrackingBenchmark> benchmark;
std::unique_ptr<ArrayGteBenchmark> arrayBenchmark;

// --- Batch size 100 ---

BENCHMARK(noTrackingSmall) {
  benchmark->runNoTracking(benchmark->smallRowVector_, kIterationsSmall);
}

BENCHMARK_RELATIVE(withTrackingSmall) {
  benchmark->runWithTracking(benchmark->smallRowVector_, kIterationsSmall);
}

BENCHMARK_RELATIVE(adaptive1PctSmall) {
  benchmark->runAdaptive1Pct(benchmark->smallRowVector_, kIterationsSmall);
}

BENCHMARK_RELATIVE(adaptiveHalfPctSmall) {
  benchmark->runAdaptiveHalfPct(benchmark->smallRowVector_, kIterationsSmall);
}

BENCHMARK_DRAW_LINE();

// --- Batch size 1,000 ---

BENCHMARK(noTrackingMedium) {
  benchmark->runNoTracking(benchmark->mediumRowVector_, kIterationsMedium);
}

BENCHMARK_RELATIVE(withTrackingMedium) {
  benchmark->runWithTracking(benchmark->mediumRowVector_, kIterationsMedium);
}

BENCHMARK_RELATIVE(adaptive1PctMedium) {
  benchmark->runAdaptive1Pct(benchmark->mediumRowVector_, kIterationsMedium);
}

BENCHMARK_RELATIVE(adaptiveHalfPctMedium) {
  benchmark->runAdaptiveHalfPct(benchmark->mediumRowVector_, kIterationsMedium);
}

BENCHMARK_DRAW_LINE();

// --- Batch size 10,000 ---

BENCHMARK(noTrackingLarge) {
  benchmark->runNoTracking(benchmark->largeRowVector_, kIterationsLarge);
}

BENCHMARK_RELATIVE(withTrackingLarge) {
  benchmark->runWithTracking(benchmark->largeRowVector_, kIterationsLarge);
}

BENCHMARK_RELATIVE(adaptive1PctLarge) {
  benchmark->runAdaptive1Pct(benchmark->largeRowVector_, kIterationsLarge);
}

BENCHMARK_RELATIVE(adaptiveHalfPctLarge) {
  benchmark->runAdaptiveHalfPct(benchmark->largeRowVector_, kIterationsLarge);
}

BENCHMARK_DRAW_LINE();

// --- array_gte: Batch size 100, 50-element arrays ---

BENCHMARK(arrayGteNoTrackingSmall) {
  arrayBenchmark->runNoTracking(
      arrayBenchmark->smallRowVector_, kIterationsSmall);
}

BENCHMARK_RELATIVE(arrayGteWithTrackingSmall) {
  arrayBenchmark->runWithTracking(
      arrayBenchmark->smallRowVector_, kIterationsSmall);
}

BENCHMARK_RELATIVE(arrayGteAdaptive1PctSmall) {
  arrayBenchmark->runAdaptive1Pct(
      arrayBenchmark->smallRowVector_, kIterationsSmall);
}

BENCHMARK_RELATIVE(arrayGteAdaptiveHalfPctSmall) {
  arrayBenchmark->runAdaptiveHalfPct(
      arrayBenchmark->smallRowVector_, kIterationsSmall);
}

BENCHMARK_DRAW_LINE();

// --- array_gte: Batch size 1,000, 50-element arrays ---

BENCHMARK(arrayGteNoTrackingMedium) {
  arrayBenchmark->runNoTracking(
      arrayBenchmark->mediumRowVector_, kIterationsMedium);
}

BENCHMARK_RELATIVE(arrayGteWithTrackingMedium) {
  arrayBenchmark->runWithTracking(
      arrayBenchmark->mediumRowVector_, kIterationsMedium);
}

BENCHMARK_RELATIVE(arrayGteAdaptive1PctMedium) {
  arrayBenchmark->runAdaptive1Pct(
      arrayBenchmark->mediumRowVector_, kIterationsMedium);
}

BENCHMARK_RELATIVE(arrayGteAdaptiveHalfPctMedium) {
  arrayBenchmark->runAdaptiveHalfPct(
      arrayBenchmark->mediumRowVector_, kIterationsMedium);
}

BENCHMARK_DRAW_LINE();

// --- array_gte: Batch size 10,000, 50-element arrays ---

BENCHMARK(arrayGteNoTrackingLarge) {
  arrayBenchmark->runNoTracking(
      arrayBenchmark->largeRowVector_, kIterationsLarge);
}

BENCHMARK_RELATIVE(arrayGteWithTrackingLarge) {
  arrayBenchmark->runWithTracking(
      arrayBenchmark->largeRowVector_, kIterationsLarge);
}

BENCHMARK_RELATIVE(arrayGteAdaptive1PctLarge) {
  arrayBenchmark->runAdaptive1Pct(
      arrayBenchmark->largeRowVector_, kIterationsLarge);
}

BENCHMARK_RELATIVE(arrayGteAdaptiveHalfPctLarge) {
  arrayBenchmark->runAdaptiveHalfPct(
      arrayBenchmark->largeRowVector_, kIterationsLarge);
}

} // namespace

int main(int argc, char* argv[]) {
  folly::Init init{&argc, &argv};
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  benchmark = std::make_unique<CpuTimeTrackingBenchmark>();
  arrayBenchmark = std::make_unique<ArrayGteBenchmark>();
  folly::runBenchmarks();
  benchmark.reset();
  arrayBenchmark.reset();
  return 0;
}
