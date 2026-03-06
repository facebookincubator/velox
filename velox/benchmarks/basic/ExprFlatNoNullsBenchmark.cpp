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

/// Benchmark comparing expression evaluation with FlatNoNulls fast path
/// enabled vs disabled via the expression.eval_flat_no_nulls config.

#include <folly/Benchmark.h>
#include <folly/init/Init.h>

#include "velox/benchmarks/ExpressionBenchmarkBuilder.h"
#include "velox/core/QueryConfig.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"

using namespace facebook;
using namespace facebook::velox;

namespace {

void addArithmeticBenchmarks(
    ExpressionBenchmarkBuilder& builder,
    const std::string& prefix) {
  for (auto vectorSize : {1'024, 4'096, 40'960, 99'999}) {
    builder
        .addBenchmarkSet(
            fmt::format("{}_arith_batch{}", prefix, vectorSize),
            ROW({"a", "b", "c", "d"}, {DOUBLE(), DOUBLE(), DOUBLE(), DOUBLE()}))
        .withFuzzerOptions(
            {.vectorSize = static_cast<size_t>(vectorSize), .nullRatio = 0.0})
        // Simple arithmetic — 1 node.
        .addExpression("add_ab", "a + b")
        // Complex arithmetic — 7 nodes.
        .addExpression("complex_7n", "(a + b) * c + (a - d) * b")
        // Deep tree — 15 nodes, depth 8 (left-skewed chain).
        .addExpression(
            "deep_15n_d8", "((((((a + b) * c - d) + a) * b - c) + d) * a - b)")
        .withIterations(1'000);
  }
}

void addComparisonBenchmarks(
    ExpressionBenchmarkBuilder& builder,
    const std::string& prefix) {
  for (auto vectorSize : {1'024, 4'096, 40'960, 99'999}) {
    builder
        .addBenchmarkSet(
            fmt::format("{}_cmp_batch{}", prefix, vectorSize),
            ROW({"a", "b"}, {DOUBLE(), DOUBLE()}))
        .withFuzzerOptions(
            {.vectorSize = static_cast<size_t>(vectorSize), .nullRatio = 0.0})
        // Comparison — 1 node.
        .addExpression("eq_ab", "a = b")
        .withIterations(1'000);
  }
}

void addConstMixedBenchmarks(
    ExpressionBenchmarkBuilder& builder,
    const std::string& prefix) {
  for (auto vectorSize : {1'024, 4'096, 40'960, 99'999}) {
    builder
        .addBenchmarkSet(
            fmt::format("{}_const_batch{}", prefix, vectorSize),
            ROW({"a", "b"}, {DOUBLE(), DOUBLE()}))
        .withFuzzerOptions(
            {.vectorSize = static_cast<size_t>(vectorSize), .nullRatio = 0.0})
        // 1 constant.
        .addExpression("const_1", "a + 1.5")
        // 2 constants.
        .addExpression("const_2", "(a + 1.5) * 2.0")
        // 3 constants mixed with columns — 7 nodes.
        .addExpression("const_3_7n", "(a + 1.5) * 2.0 + (a - 3.0) * b")
        .withIterations(1'000);
  }
}

/// Extends ExpressionBenchmarkBuilder to allow setting query config.
class ConfigurableBenchmarkBuilder : public ExpressionBenchmarkBuilder {
 public:
  void setConfig(const std::string& key, const std::string& value) {
    queryCtx_->testingOverrideConfigUnsafe({{key, value}});
  }
};

} // namespace

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  functions::prestosql::registerAllScalarFunctions();

  // Fast path ON (default).
  ConfigurableBenchmarkBuilder fastPathOn;
  addArithmeticBenchmarks(fastPathOn, "on");
  addComparisonBenchmarks(fastPathOn, "on");
  addConstMixedBenchmarks(fastPathOn, "on");

  // Fast path OFF via config.
  ConfigurableBenchmarkBuilder fastPathOff;
  fastPathOff.setConfig(core::QueryConfig::kExprEvalFlatNoNulls, "false");
  addArithmeticBenchmarks(fastPathOff, "off");
  addComparisonBenchmarks(fastPathOff, "off");
  addConstMixedBenchmarks(fastPathOff, "off");

  fastPathOn.registerBenchmarks();
  fastPathOff.registerBenchmarks();
  folly::runBenchmarks();
  return 0;
}
