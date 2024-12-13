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
#include "velox/functions/sparksql/Factorial.h"
#include "velox/expression/ConstantExpr.h"
#include "velox/expression/VectorFunction.h"
#include <limits>
#include <iostream>

namespace facebook::velox::functions::sparksql {

namespace {

/**
 * Computes the factorial of integers in the range [0...20]
 *
 * Returns NULL for inputs which are outside the range [0...20].
 * Leverages a lookup table for O(1) computation, similar to Spark JVM.
 */
class Factorial : public exec::VectorFunction {
 public:
  Factorial() = default;

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const override {

    context.ensureWritable(rows, BIGINT(), result);
    auto* flatResult = result->asFlatVector<int64_t>();

    exec::DecodedArgs decodedArgs(rows, args, context);
    auto* inputVector = decodedArgs.at(0);

    rows.applyToSelected([&](vector_size_t row) {
      if (inputVector->isNullAt(row)) {
        flatResult->setNull(row, true);
      } else {
        int32_t value = inputVector->valueAt<int32_t>(row);
        if (value < LOWER_BOUND || value > UPPER_BOUND) {
          flatResult->setNull(row, true);
        } else {
          flatResult->set(row, kFactorials[value]);
        }
      }
    });
  }

 private:
  static constexpr int64_t LOWER_BOUND = 0;
  static constexpr int64_t UPPER_BOUND = 20;
  static constexpr int64_t MAX_INT64 = std::numeric_limits<int64_t>::max();

  static constexpr int64_t kFactorials[21] = {
    1,
    1,
    2,
    6,
    24,
    120,
    720,
    5040,
    40320,
    362880,
    3628800,
    39916800,
    479001600,
    6227020800L,
    87178291200L,
    1307674368000L,
    20922789888000L,
    355687428096000L,
    6402373705728000L,
    121645100408832000L,
    2432902008176640000L
  };
};
} // namespace

TypePtr FactorialCallToSpecialForm::resolveType(
    const std::vector<TypePtr>&) {
  return BIGINT();
}

exec::ExprPtr FactorialCallToSpecialForm::constructSpecialForm(
    const TypePtr& type,
    std::vector<exec::ExprPtr>&& args,
    bool trackCpuUsage,
    const core::QueryConfig& config) {
  auto numArgs = args.size();

  VELOX_USER_CHECK_EQ(
      numArgs,
      1,
      "factorial requires exactly 1 argument, but got {}.",
      numArgs);
  VELOX_USER_CHECK(
      args[0]->type()->isInteger(),
      "The argument of factorial must be an integer.");

  auto factorial = std::make_shared<Factorial>();
  return std::make_shared<exec::Expr>(
      type,
      std::move(args),
      std::move(factorial),
      exec::VectorFunctionMetadataBuilder().defaultNullBehavior(false).build(),
      "factorial",
      trackCpuUsage);
}
} // namespace facebook::velox::functions::sparksql
