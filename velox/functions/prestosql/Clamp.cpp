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
#include "velox/functions/prestosql/Clamp.h"
#include <folly/Likely.h>
#include "velox/expression/Expr.h"
#include "velox/expression/VectorFunction.h"

namespace facebook::velox::functions {
namespace {

template <typename T>
class ClampFunction : public exec::VectorFunction {
 public:
  ClampFunction() : constLowHigh_{false}, low_{}, high_{} {}

  ClampFunction(T low, T high) : constLowHigh_{true}, low_{low}, high_{high} {}

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& resultType,
      exec::EvalCtx* context,
      VectorPtr* result) const override {
    if (constLowHigh_ && args[0]->encoding() == VectorEncoding::Simple::FLAT) {
      // Fast path for (flat, const, const).

      // We cannot perform this check in the constructor because the error
      // should not occur unless the function was applied to at least one row.
      VELOX_USER_CHECK_LE(low_, high_, "Low > high in clamp.");

      auto flatValues = args[0]->asUnchecked<FlatVector<T>>();
      auto rawValues = flatValues->mutableRawValues();

      // Check if input can be reused for results.
      T* rawResults;
      if (!(*result) && BaseVector::isReusableFlatVector(args[0])) {
        rawResults = rawValues;
        *result = std::move(args[0]);
      } else {
        rawResults = prepareResult(rows, resultType, context, result);
      }

      context->applyToSelectedNoThrow(rows, [&](auto row) {
        rawResults[row] = std::clamp(rawValues[row], low_, high_);
      });
    } else {
      exec::DecodedArgs decodedArgs(rows, args, context);

      auto values = decodedArgs.at(0);
      auto lows = decodedArgs.at(1);
      auto highs = decodedArgs.at(2);

      auto rawResults = prepareResult(rows, resultType, context, result);

      context->applyToSelectedNoThrow(rows, [&](auto row) {
        auto low = lows->valueAt<T>(row);
        auto high = highs->valueAt<T>(row);

        VELOX_USER_CHECK_LE(low, high, "Low > high in clamp.");
        rawResults[row] = std::clamp(values->valueAt<T>(row), low, high);
      });
    }
  }

 private:
  T* prepareResult(
      const SelectivityVector& rows,
      const TypePtr& resultType,
      exec::EvalCtx* context,
      VectorPtr* result) const {
    BaseVector::ensureWritable(rows, resultType, context->pool(), result);
    (*result)->clearNulls(rows);
    return (*result)->asUnchecked<FlatVector<T>>()->mutableRawValues();
  }

  const bool constLowHigh_;
  const T low_;
  const T high_;
};

std::vector<std::shared_ptr<exec::FunctionSignature>> clampSignatures() {
  std::vector<std::shared_ptr<exec::FunctionSignature>> signatures;
  for (auto type :
       {"tinyint", "smallint", "integer", "bigint", "real", "double"}) {
    signatures.push_back(exec::FunctionSignatureBuilder()
                             .returnType(type)
                             .argumentType(type)
                             .argumentType(type)
                             .argumentType(type)
                             .build());
  }
  return signatures;
}

template <typename T>
std::shared_ptr<exec::VectorFunction> makeClampTyped(
    const VectorPtr& lowConst,
    const VectorPtr& highConst) {
  // Constant folding is not sophisticated enough to detect a function call with
  // some inputs being null.
  if (lowConst && highConst && !lowConst->isNullAt(0) &&
      !highConst->isNullAt(0)) {
    auto low = lowConst->as<SimpleVector<T>>()->valueAt(0);
    auto high = highConst->as<SimpleVector<T>>()->valueAt(0);
    return std::make_shared<ClampFunction<T>>(low, high);
  } else {
    return std::make_shared<ClampFunction<T>>();
  }
}

std::shared_ptr<exec::VectorFunction> makeClamp(
    const std::string& /*name*/,
    const std::vector<exec::VectorFunctionArg>& inputArgs) {
  auto typeKind = inputArgs[0].type->kind();
  auto& low = inputArgs[1].constantValue;
  auto& high = inputArgs[2].constantValue;
  switch (typeKind) {
    case TypeKind::TINYINT:
      return makeClampTyped<int8_t>(low, high);
    case TypeKind::SMALLINT:
      return makeClampTyped<int16_t>(low, high);
    case TypeKind::INTEGER:
      return makeClampTyped<int32_t>(low, high);
    case TypeKind::BIGINT:
      return makeClampTyped<int64_t>(low, high);
    case TypeKind::REAL:
      return makeClampTyped<float>(low, high);
    case TypeKind::DOUBLE:
      return makeClampTyped<double>(low, high);
    default:
      VELOX_UNREACHABLE();
  }
}
} // namespace

void registerClamp(const std::string& name) {
  exec::registerStatefulVectorFunction(name, clampSignatures(), makeClamp);
}
} // namespace facebook::velox::functions
