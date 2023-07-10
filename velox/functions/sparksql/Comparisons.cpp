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
#include "velox/functions/sparksql/LeastGreatest.h"

#include "velox/expression/EvalCtx.h"
#include "velox/expression/Expr.h"
#include "velox/functions/sparksql/Comparisons.h"
#include "velox/type/Type.h"

namespace facebook::velox::functions::sparksql {
namespace {

template <typename Cmp, TypeKind kind>
class ComparisonFunction final : public exec::VectorFunction {
  using T = typename TypeTraits<kind>::NativeType;

  bool supportsFlatNoNullsFastPath() const override {
    return true;
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    context.ensureWritable(rows, BOOLEAN(), result);
    auto* flatResult = result->asFlatVector<bool>();
    const Cmp cmp;
    if (args[0]->isFlatEncoding() && args[1]->isFlatEncoding()) {
      // Fast path for (flat, flat).
      auto flatA = args[0]->asUnchecked<FlatVector<T>>();
      auto rawA = flatA->mutableRawValues();
      auto flatB = args[1]->asUnchecked<FlatVector<T>>();
      auto rawB = flatB->mutableRawValues();
      rows.applyToSelected(
          [&](vector_size_t i) { flatResult->set(i, cmp(rawA[i], rawB[i])); });
    } else if (args[0]->isConstantEncoding() && args[1]->isFlatEncoding()) {
      // Fast path for (const, flat).
      auto constant = args[0]->asUnchecked<ConstantVector<T>>()->valueAt(0);
      auto flatValues = args[1]->asUnchecked<FlatVector<T>>();
      auto rawValues = flatValues->mutableRawValues();
      rows.applyToSelected([&](vector_size_t i) {
        flatResult->set(i, cmp(constant, rawValues[i]));
      });
    } else if (args[0]->isFlatEncoding() && args[1]->isConstantEncoding()) {
      // Fast path for (flat, const).
      auto flatValues = args[0]->asUnchecked<FlatVector<T>>();
      auto constant = args[1]->asUnchecked<ConstantVector<T>>()->valueAt(0);
      auto rawValues = flatValues->mutableRawValues();
      rows.applyToSelected([&](vector_size_t i) {
        flatResult->set(i, cmp(rawValues[i], constant));
      });
    } else {
      // Fast path if one or more arguments are encoded.
      exec::DecodedArgs decodedArgs(rows, args, context);
      auto decoded0 = decodedArgs.at(0);
      auto decoded1 = decodedArgs.at(1);
      rows.applyToSelected([&](vector_size_t i) {
        flatResult->set(
            i, cmp(decoded0->valueAt<T>(i), decoded1->valueAt<T>(i)));
      });
    }
  }
};

template <template <typename> class Cmp>
std::shared_ptr<exec::VectorFunction> makeImpl(
    const std::string& functionName,
    const std::vector<exec::VectorFunctionArg>& args) {
  VELOX_CHECK_EQ(args.size(), 2);
  for (size_t i = 1; i < args.size(); i++) {
    VELOX_CHECK(*args[i].type == *args[0].type);
  }
  switch (args[0].type->kind()) {
#define SCALAR_CASE(kind)                            \
  case TypeKind::kind:                               \
    return std::make_shared<ComparisonFunction<      \
        Cmp<TypeTraits<TypeKind::kind>::NativeType>, \
        TypeKind::kind>>();
    SCALAR_CASE(BOOLEAN)
    SCALAR_CASE(TINYINT)
    SCALAR_CASE(SMALLINT)
    SCALAR_CASE(INTEGER)
    SCALAR_CASE(BIGINT)
    SCALAR_CASE(HUGEINT)
    SCALAR_CASE(REAL)
    SCALAR_CASE(DOUBLE)
    SCALAR_CASE(VARCHAR)
    SCALAR_CASE(VARBINARY)
    SCALAR_CASE(TIMESTAMP)
#undef SCALAR_CASE
    default:
      VELOX_NYI(
          "{} does not support arguments of type {}",
          functionName,
          args[0].type->kind());
  }
}

} // namespace

std::shared_ptr<exec::VectorFunction> makeEqualTo(
    const std::string& functionName,
    const std::vector<exec::VectorFunctionArg>& args,
    const core::QueryConfig& /*config*/) {
  return makeImpl<Equal>(functionName, args);
}

std::shared_ptr<exec::VectorFunction> makeLessThan(
    const std::string& functionName,
    const std::vector<exec::VectorFunctionArg>& args,
    const core::QueryConfig& /*config*/) {
  return makeImpl<Less>(functionName, args);
}

std::shared_ptr<exec::VectorFunction> makeGreaterThan(
    const std::string& functionName,
    const std::vector<exec::VectorFunctionArg>& args,
    const core::QueryConfig& /*config*/) {
  return makeImpl<Greater>(functionName, args);
}

std::shared_ptr<exec::VectorFunction> makeLessThanOrEqual(
    const std::string& functionName,
    const std::vector<exec::VectorFunctionArg>& args,
    const core::QueryConfig& /*config*/) {
  return makeImpl<LessOrEqual>(functionName, args);
}

std::shared_ptr<exec::VectorFunction> makeGreaterThanOrEqual(
    const std::string& functionName,
    const std::vector<exec::VectorFunctionArg>& args,
    const core::QueryConfig& /*config*/) {
  return makeImpl<GreaterOrEqual>(functionName, args);
}
} // namespace facebook::velox::functions::sparksql
