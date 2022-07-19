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
#include "velox/expression/VectorFunction.h"
#include "velox/functions/Macros.h"
#include "velox/functions/Registerer.h"
#include "velox/functions/lib/LambdaFunctionUtil.h"
#include "velox/functions/lib/benchmarks/FunctionBenchmarkBase.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::functions;

namespace {

///
/// Implements the array_sum function.
/// See documentation at https://prestodb.io/docs/current/functions/array.html
///
template <typename IT, typename OT>
class ArraySumFunction : public exec::VectorFunction {
 public:
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args, // Not using const ref so we can reuse args
      const TypePtr& outputType,
      exec::EvalCtx* context,
      VectorPtr* result) const override {
    // Acquire the array elements vector.
    auto arrayVector = args[0]->as<ArrayVector>();
    VELOX_CHECK(arrayVector);
    auto elementsVector = arrayVector->elements();
    auto elementsRows =
        toElementRows(elementsVector->size(), rows, arrayVector);
    exec::LocalDecodedVector elements(context, *elementsVector, elementsRows);
    vector_size_t numRows = arrayVector->size();

    // Prepare result vector for writing
    BaseVector::ensureWritable(rows, outputType, context->pool(), result);
    auto resultValues = (*result)->template asFlatVector<OT>();

    rows.template applyToSelected([&](vector_size_t row) {
      if (arrayVector->isNullAt(row)) {
        resultValues->setNull(row, true);
      } else {
        int start = arrayVector->offsetAt(row);
        int end = start + arrayVector->sizeAt(row);

        OT sum = 0;
        for (; start < end; start++) {
          if (!elements->isNullAt(start)) {
            sum += elements->template valueAt<IT>(start);
          }
        }
        resultValues->set(row, sum);
      }
    });
  }
};

// Validate number of parameters and types.
void validateType(const std::vector<exec::VectorFunctionArg>& inputArgs) {
  VELOX_USER_CHECK_EQ(
      inputArgs.size(), 1, "array_sum requires exactly one parameter");

  auto arrayType = inputArgs.front().type;
  VELOX_USER_CHECK_EQ(
      arrayType->kind(),
      TypeKind::ARRAY,
      "array_sum requires argument of type ARRAY");

  auto valueTypeKind = inputArgs.front().type->childAt(0)->kind();
  bool isCoercibleToDouble = valueTypeKind == TypeKind::TINYINT ||
      valueTypeKind == TypeKind::SMALLINT ||
      valueTypeKind == TypeKind::INTEGER || valueTypeKind == TypeKind::BIGINT ||
      valueTypeKind == TypeKind::REAL || valueTypeKind == TypeKind::DOUBLE;
  auto errorMessage =
      std::string("Invalid value type: ") + mapTypeKindToName(valueTypeKind);
  VELOX_USER_CHECK_EQ(isCoercibleToDouble, true, "{}", errorMessage);
}

// Create function.
std::shared_ptr<exec::VectorFunction> create(
    const std::string& /* name */,
    const std::vector<exec::VectorFunctionArg>& inputArgs) {
  validateType(inputArgs);
  auto elementType = inputArgs.front().type->childAt(0);

  switch (elementType->kind()) {
    case TypeKind::TINYINT: {
      return std::make_shared<ArraySumFunction<
          TypeTraits<TypeKind::TINYINT>::NativeType,
          int64_t>>();
    }
    case TypeKind::SMALLINT: {
      return std::make_shared<ArraySumFunction<
          TypeTraits<TypeKind::SMALLINT>::NativeType,
          int64_t>>();
    }
    case TypeKind::INTEGER: {
      return std::make_shared<ArraySumFunction<
          TypeTraits<TypeKind::INTEGER>::NativeType,
          int64_t>>();
    }
    case TypeKind::BIGINT: {
      return std::make_shared<ArraySumFunction<
          TypeTraits<TypeKind::BIGINT>::NativeType,
          int64_t>>();
    }
    case TypeKind::REAL: {
      return std::make_shared<
          ArraySumFunction<TypeTraits<TypeKind::REAL>::NativeType, double>>();
    }
    case TypeKind::DOUBLE: {
      return std::make_shared<
          ArraySumFunction<TypeTraits<TypeKind::DOUBLE>::NativeType, double>>();
    }
    default: {
      VELOX_FAIL("Unsupported Type")
    }
  }
}

// Define function signature.
// array(T1) -> T2 where T1 must be coercible to bigint or double, and
// T2 is bigint or double
std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
  static const std::map<std::string, std::string> s = {
      {"tinyint", "bigint"},
      {"smallint", "bigint"},
      {"integer", "bigint"},
      {"bigint", "bigint"},
      {"real", "double"},
      {"double", "double"}};
  std::vector<std::shared_ptr<exec::FunctionSignature>> signatures;
  signatures.reserve(s.size());
  for (const auto& [argType, returnType] : s) {
    signatures.emplace_back(exec::FunctionSignatureBuilder()
                                .returnType(returnType)
                                .argumentType(fmt::format("array({})", argType))
                                .build());
  }
  return signatures;
}

// Register function.
VELOX_DECLARE_STATEFUL_VECTOR_FUNCTION(udf_array_sum, signatures(), create);

class ArraySumBenchmark : public functions::test::FunctionBenchmarkBase {
 public:
  ArraySumBenchmark() : FunctionBenchmarkBase() {
    functions::prestosql::registerArrayFunctions();
    functions::prestosql::registerGeneralFunctions();

    VELOX_REGISTER_VECTOR_FUNCTION(udf_array_sum, "array_sum_alt");
  }

  void runInteger(const std::string& functionName) {
    folly::BenchmarkSuspender suspender;
    vector_size_t size = 10'000;
    auto arrayVector = vectorMaker_.arrayVector<int32_t>(
        size,
        [](auto row) { return row % 5; },
        [](auto row) { return row % 23; });

    auto rowVector = vectorMaker_.rowVector({arrayVector});
    auto exprSet = compileExpression(
        fmt::format("{}(c0)", functionName), rowVector->type());
    suspender.dismiss();

    doRun(exprSet, rowVector);
  }

  void doRun(ExprSet& exprSet, const RowVectorPtr& rowVector) {
    int cnt = 0;
    for (auto i = 0; i < 100; i++) {
      cnt += evaluate(exprSet, rowVector)->size();
    }
    folly::doNotOptimizeAway(cnt);
  }
};

BENCHMARK(vectorSimpleFunction) {
  ArraySumBenchmark benchmark;
  benchmark.runInteger("array_sum");
}

BENCHMARK_RELATIVE(vectorFunctionInteger) {
  ArraySumBenchmark benchmark;
  benchmark.runInteger("array_sum_alt");
}

} // namespace

int main(int /*argc*/, char** /*argv*/) {
  folly::runBenchmarks();
  return 0;
}