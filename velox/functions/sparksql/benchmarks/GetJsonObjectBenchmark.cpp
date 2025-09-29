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

#include "velox/core/Expressions.h"
#include "velox/functions/lib/benchmarks/FunctionBenchmarkBase.h"
#include "velox/functions/sparksql/registration/Register.h"

using namespace facebook::velox;
namespace {
class GetJsonObjectBenchmark : public functions::test::FunctionBenchmarkBase {
 public:
  GetJsonObjectBenchmark() : FunctionBenchmarkBase() {
    functions::sparksql::registerFunctions("");
  }

  void run(
      const StringView& json,
      const StringView& jsonPath,
      const bool constPath = true) {
    folly::BenchmarkSuspender suspender;
    auto rowVector = vectorMaker_.rowVector(
        {"c0", "c1"},
        {vectorMaker_.flatVector<StringView>(
             1'000, [&](vector_size_t /*row*/) { return json; }),
         vectorMaker_.flatVector<StringView>(
             1'000, [&](vector_size_t /*row*/) { return jsonPath; })});

    core::TypedExprPtr pathExpr;
    if (constPath) {
      pathExpr = std::make_shared<core::ConstantTypedExpr>(VARCHAR(), jsonPath);
    } else {
      pathExpr = std::make_shared<core::FieldAccessTypedExpr>(VARCHAR(), "c1");
    }
    auto typedExpr = std::make_shared<const core::CallTypedExpr>(
        VARCHAR(),
        "get_json_object",
        std::make_shared<core::FieldAccessTypedExpr>(VARCHAR(), "c0"),
        pathExpr);
    exec::ExprSet exprSet({typedExpr}, &execCtx_);
    suspender.dismiss();

    for (auto i = 0; i < 1000; i++) {
      evaluate(exprSet, rowVector)->size();
    }
  }
};

BENCHMARK(number_depth1) {
  GetJsonObjectBenchmark benchmark;
  benchmark.run(StringView(R"({"a": 100})"), StringView("$.a"));
}

BENCHMARK(number_depth1_variable_path) {
  GetJsonObjectBenchmark benchmark;
  benchmark.run(StringView(R"({"a": 100})"), StringView("$.a"), false);
}

BENCHMARK(number_depth2) {
  GetJsonObjectBenchmark benchmark;
  benchmark.run(StringView(R"({"a": {"b": 123}})"), StringView("$.a.b"));
}

BENCHMARK(object_depth1_1col) {
  GetJsonObjectBenchmark benchmark;
  benchmark.run(StringView(R"({"a": {"b": 123}})"), StringView("$.a"));
}

BENCHMARK(object_depth3_4col) {
  GetJsonObjectBenchmark benchmark;
  benchmark.run(
      StringView(R"({"a": {"b": {"c": {"d": 1, "e": 2, "f": 3, "g": 4}}}})"),
      StringView("$.a.b.c"));
}

BENCHMARK(array_depth1_1element) {
  GetJsonObjectBenchmark benchmark;
  benchmark.run(StringView(R"({"a": [1]})"), StringView("$.a"));
}

BENCHMARK(array_depth4_20element) {
  GetJsonObjectBenchmark benchmark;
  benchmark.run(
      StringView(
          R"({"a": {"b": {"c": {"d": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]}}}})"),
      StringView("$.a.b.c.d"));
}
} // namespace

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  folly::runBenchmarks();
  return 0;
}
