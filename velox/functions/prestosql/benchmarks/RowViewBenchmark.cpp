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
#include "velox/expression/ComplexViewTypes.h"
#include "velox/expression/EvalCtx.h"
#include "velox/expression/VectorFunction.h"
#include "velox/functions/Macros.h"
#include "velox/functions/Registerer.h"
#include "velox/functions/lib/benchmarks/FunctionBenchmarkBase.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"

namespace facebook::velox {
namespace {
using namespace facebook::velox::exec;
class RowSum : public exec::VectorFunction {
 public:
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /* outputType */,
      exec::EvalCtx* context,
      VectorPtr* result) const override {
    exec::DecodedArgs decodedArgs(rows, args, context);
    auto* rowsVector = decodedArgs.at(0)->base()->template as<RowVector>();
    std::vector<DecodedVector> childrenDecoders(rowsVector->childrenSize());

    VELOX_CHECK(rowsVector->childrenSize() > 0);
    exec::LocalSelectivityVector childRows(
        context, rowsVector->childAt(0)->size());

    for (auto i = 0; i < rowsVector->childrenSize(); i++) {
      childrenDecoders[i].decode(*rowsVector->childAt(i), *childRows);
    }

    BaseVector::ensureWritable(rows, BIGINT(), context->pool(), result);

    auto flatResults = result->get()->asFlatVector<int64_t>();
    rows.applyToSelected([&](vector_size_t row) {
      int64_t sum = 0;

      auto index = decodedArgs.at(0)->index(row);
      for (const auto& decoded : childrenDecoders) {
        sum += decoded.valueAt<int64_t>(index);
      }
      flatResults->set(row, sum);
    });
  }
};

// Use vector reader to read args.
class RowSumWithReader : public exec::VectorFunction {
 public:
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /* outputType */,
      exec::EvalCtx* context,
      VectorPtr* result) const override {
    exec::DecodedArgs decodedArgs(rows, args, context);

    BaseVector::ensureWritable(rows, BIGINT(), context->pool(), result);

    auto reader = VectorReader<Row<int64_t, int64_t, int64_t, int64_t>>(
        decodedArgs.at(0));

    auto flatResults = result->get()->asFlatVector<int64_t>();

    rows.applyToSelected([&](vector_size_t row) {
      int64_t sum = 0;

      auto rowView = reader[row];
      sum += rowView.template at<0>().value();
      sum += rowView.template at<1>().value();
      sum += rowView.template at<2>().value();
      sum += rowView.template at<3>().value();
      flatResults->set(row, sum);
    });
  }
};

template <typename T>
struct RowSumSimple {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  FOLLY_ALWAYS_INLINE bool call(
      int64_t& out,
      const arg_type<Row<int64_t, int64_t, int64_t, int64_t>>& input) {
    int64_t sum = 0;
    sum += input.template at<0>().value();
    sum += input.template at<1>().value();
    sum += input.template at<2>().value();
    sum += input.template at<3>().value();
    out = sum;
    return true;
  }
};

class RowViewBenchmark : public functions::test::FunctionBenchmarkBase {
 public:
  RowViewBenchmark() : FunctionBenchmarkBase() {
    registerFunction<
        RowSumSimple,
        int64_t,
        Row<int64_t, int64_t, int64_t, int64_t>>({"f_simple"});

    facebook::velox::exec::registerVectorFunction(
        "f_vector",
        {exec::FunctionSignatureBuilder()
             .returnType("bigint")
             .argumentType("row(bigint, bigint, bigint, bigint)")
             .build()},
        std::make_unique<RowSum>());

    facebook::velox::exec::registerVectorFunction(
        "f_vector_reader",
        {exec::FunctionSignatureBuilder()
             .returnType("bigint")
             .argumentType("row(bigint, bigint, bigint, bigint)")
             .build()},
        std::make_unique<RowSumWithReader>());
  }

  RowVectorPtr makeData() {
    const vector_size_t size = 1'000;
    auto vector1 = vectorMaker_.flatVector(std::vector<int64_t>(size, 1));
    auto vector2 = vectorMaker_.flatVector(std::vector<int64_t>(size, 2));
    auto vector3 = vectorMaker_.flatVector(std::vector<int64_t>(size, 3));
    auto vector4 = vectorMaker_.flatVector(std::vector<int64_t>(size, 4));

    return vectorMaker_.rowVector(
        {vectorMaker_.rowVector({vector1, vector2, vector3, vector4})});
  }

  size_t run(const std::string& functionName) {
    folly::BenchmarkSuspender suspender;
    auto rowVector = makeData();
    auto exprSet = compileExpression(
        fmt::format("{}(c0)", functionName), rowVector->type());
    suspender.dismiss();

    return doRun(exprSet, rowVector);
  }

  size_t doRun(ExprSet& exprSet, const RowVectorPtr& rowVector) {
    int cnt = 0;
    for (auto i = 0; i < 100; i++) {
      cnt += evaluate(exprSet, rowVector)->size();
    }
    folly::doNotOptimizeAway(cnt);
    return cnt;
  }

  bool hasSameResults(
      ExprSet& expr1,
      ExprSet& expr2,
      const RowVectorPtr& rowVector) {
    auto result1 = evaluate(expr1, rowVector);
    auto result2 = evaluate(expr2, rowVector);
    auto flatResults1 = result1->asFlatVector<int64_t>();
    auto flatResults2 = result2->asFlatVector<int64_t>();

    if (flatResults1->size() != flatResults2->size()) {
      return false;
    }

    for (auto i = 0; i < flatResults1->size(); i++) {
      if (flatResults1->valueAt(i) != flatResults2->valueAt(i)) {
        return false;
      }
    }

    return true;
  }

  bool test() {
    auto rowVector = makeData();
    auto exprSet1 = compileExpression("f_vector(c0)", rowVector->type());
    auto exprSet2 = compileExpression("f_simple(c0)", rowVector->type());
    auto exprSet3 = compileExpression("f_vector_reader(c0)", rowVector->type());
    return hasSameResults(exprSet1, exprSet2, rowVector) &&
        hasSameResults(exprSet1, exprSet3, rowVector);
  }
};

BENCHMARK_MULTI(RowViewSimple) {
  RowViewBenchmark benchmark;
  return benchmark.run("f_simple");
}

BENCHMARK_MULTI(RowViewVector) {
  RowViewBenchmark benchmark;
  return benchmark.run("f_vector");
}

BENCHMARK_MULTI(RowViewVectorWithReader) {
  RowViewBenchmark benchmark;
  return benchmark.run("f_vector_reader");
}
} // namespace
} // namespace facebook::velox

int main(int /*argc*/, char** /*argv*/) {
  facebook::velox::RowViewBenchmark benchmark;
  if (benchmark.test()) {
    folly::runBenchmarks();
  } else {
    VELOX_UNREACHABLE("tests failed");
  }
  return 0;
}
