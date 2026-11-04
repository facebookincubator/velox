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

#include <random>

#include "velox/core/Expressions.h"
#include "velox/expression/Expr.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"

// Measures the time to compile very large expressions. The expressions have the
// shape of gradient boosted tree models (e.g. LightGBM) translated to SQL: a
// left-deep sum of many decision trees, each a nested IF over comparisons of
// input features with constants. Each tree of depth 5 has 156 nodes, so 1,600
// trees make a single expression of about 250K nodes, nested 1,600 levels deep.

using namespace facebook::velox;

namespace {

class ExprCompileBenchmark {
 public:
  ExprCompileBenchmark() {
    functions::prestosql::registerAllScalarFunctions();
  }

  // Returns a sum of 'numTrees' decision trees. Uses a fixed seed so every
  // call returns the same expression.
  core::TypedExprPtr makeModel(int numTrees) {
    // Add up the trees left-deep, as `tree1 + tree2 + ...` parses.
    std::mt19937 rng(42);
    auto sum = makeTree(kTreeDepth, rng);
    for (auto i = 1; i < numTrees; ++i) {
      sum = std::make_shared<core::CallTypedExpr>(
          DOUBLE(),
          std::vector<core::TypedExprPtr>{sum, makeTree(kTreeDepth, rng)},
          "plus");
    }
    return sum;
  }

  std::unique_ptr<exec::ExprSet> compile(
      const core::TypedExprPtr& expr,
      bool constantFolding) {
    return std::make_unique<exec::ExprSet>(
        std::vector<core::TypedExprPtr>{expr}, &execCtx_, constantFolding);
  }

 private:
  static constexpr int kTreeDepth{5};
  static constexpr int kNumFeatures{200};

  // Returns IF(f_i <= c, <left subtree>, <right subtree>) nested 'depth'
  // levels, with constant leaves.
  core::TypedExprPtr makeTree(int depth, std::mt19937& rng) {
    // Leaves are constant scores.
    std::uniform_real_distribution<double> value(-1.0, 1.0);
    if (depth == 0) {
      return std::make_shared<core::ConstantTypedExpr>(
          DOUBLE(), Variant(value(rng)));
    }

    // Inner nodes split on a feature compared with a threshold.
    auto feature = std::make_shared<core::FieldAccessTypedExpr>(
        DOUBLE(), fmt::format("f{}", rng() % kNumFeatures));
    auto threshold = std::make_shared<core::ConstantTypedExpr>(
        DOUBLE(), Variant(value(rng)));
    auto condition = std::make_shared<core::CallTypedExpr>(
        BOOLEAN(), std::vector<core::TypedExprPtr>{feature, threshold}, "lte");
    return std::make_shared<core::CallTypedExpr>(
        DOUBLE(),
        std::vector<core::TypedExprPtr>{
            condition, makeTree(depth - 1, rng), makeTree(depth - 1, rng)},
        "if");
  }

  std::shared_ptr<core::QueryCtx> queryCtx_{core::QueryCtx::create()};
  std::shared_ptr<memory::MemoryPool> pool_{
      memory::memoryManager()->addLeafPool()};
  core::ExecCtx execCtx_{pool_.get(), queryCtx_.get()};
};

std::unique_ptr<ExprCompileBenchmark> benchmark;

// Builds a new expression for every iteration, as compiling a deserialized
// plan would, so that nothing computed on the expression carries over between
// iterations. Only compilation is timed.
void compileModel(uint32_t iterations, int numTrees, bool constantFolding) {
  for (auto i = 0; i < iterations; ++i) {
    // Build a new expression, untimed.
    core::TypedExprPtr model;
    BENCHMARK_SUSPEND {
      model = benchmark->makeModel(numTrees);
    }

    // Compile it.
    auto exprSet = benchmark->compile(model, constantFolding);

    // Free the compiled and input expressions, untimed.
    BENCHMARK_SUSPEND {
      exprSet.reset();
      model.reset();
    }
  }
}

BENCHMARK_NAMED_PARAM(compileModel, trees100, 100, false)
BENCHMARK_RELATIVE_NAMED_PARAM(compileModel, trees100_folding, 100, true)
BENCHMARK_NAMED_PARAM(compileModel, trees400, 400, false)
BENCHMARK_RELATIVE_NAMED_PARAM(compileModel, trees400_folding, 400, true)
BENCHMARK_NAMED_PARAM(compileModel, trees1600, 1'600, false)
BENCHMARK_RELATIVE_NAMED_PARAM(compileModel, trees1600_folding, 1'600, true)

} // namespace

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  benchmark = std::make_unique<ExprCompileBenchmark>();
  folly::runBenchmarks();
  benchmark.reset();
  return 0;
}
