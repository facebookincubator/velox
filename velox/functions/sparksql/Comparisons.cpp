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

    exec::DecodedArgs decodedArgs(rows, args, context);
    auto decodedArg0 = decodedArgs.at(0);
    auto decodedArg1 = decodedArgs.at(1);
    if (decodedArg0->isIdentityMapping() && decodedArg1->isConstantMapping()) {
      rows.applyToSelected([&](auto i) {
        flatResult->set(
            i, cmp(decodedArg0->valueAt<T>(i), decodedArg1->valueAt<T>(0)));
      });
    } else if (
        decodedArg0->isConstantMapping() && decodedArg1->isIdentityMapping()) {
      rows.applyToSelected([&](auto i) {
        flatResult->set(
            i, cmp(decodedArg0->valueAt<T>(0), decodedArg1->valueAt<T>(i)));
      });
    } else {
      rows.applyToSelected([&](auto i) {
        flatResult->set(
            i, cmp(decodedArg0->valueAt<T>(i), decodedArg1->valueAt<T>(i)));
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

template <TypeKind kind>
void applyTyped(
    const SelectivityVector& rows,
    std::vector<VectorPtr>& args,
    DecodedVector* decoded0,
    DecodedVector* decoded1,
    exec::EvalCtx& context,
    FlatVector<bool>* flatResult) {
  using T = typename TypeTraits<kind>::NativeType;

  Equal<T> equal;
  if (!args[0]->mayHaveNulls() && !args[1]->mayHaveNulls()) {
    // When there is no nulls, it reduces to normal equality function
    rows.applyToSelected([&](vector_size_t i) {
      flatResult->set(
          i, equal(decoded0->valueAt<T>(i), decoded1->valueAt<T>(i)));
    });
  } else {
    // (isnull(a) AND isnull(b)) || (a == b)
    // When DecodedVector::nulls() is null it means there are no nulls.
    auto* rawNulls0 = decoded0->nulls();
    auto* rawNulls1 = decoded1->nulls();
    rows.applyToSelected([&](vector_size_t i) {
      auto isNull0 = rawNulls0 && bits::isBitNull(rawNulls0, i);
      auto isNull1 = rawNulls1 && bits::isBitNull(rawNulls1, i);
      flatResult->set(
          i,
          (isNull0 || isNull1)
              ? isNull0 && isNull1
              : equal(decoded0->valueAt<T>(i), decoded1->valueAt<T>(i)));
    });
  }
}

template <>
void applyTyped<TypeKind::ARRAY>(
    const SelectivityVector& /* rows */,
    std::vector<VectorPtr>& /* args */,
    DecodedVector* /* decoded0 */,
    DecodedVector* /* decoded1 */,
    exec::EvalCtx& /* context */,
    FlatVector<bool>* /* flatResult */) {
  VELOX_NYI("equalnullsafe does not support arrays.");
}

template <>
void applyTyped<TypeKind::MAP>(
    const SelectivityVector& /* rows */,
    std::vector<VectorPtr>& /* args */,
    DecodedVector* /* decoded0 */,
    DecodedVector* /* decoded1 */,
    exec::EvalCtx& /* context */,
    FlatVector<bool>* /* flatResult */) {
  VELOX_NYI("equalnullsafe does not support maps.");
}

template <>
void applyTyped<TypeKind::ROW>(
    const SelectivityVector& /* rows */,
    std::vector<VectorPtr>& /* args */,
    DecodedVector* /* decoded0 */,
    DecodedVector* /* decoded1 */,
    exec::EvalCtx& /* context */,
    FlatVector<bool>* /* flatResult */) {
  VELOX_NYI("equalnullsafe does not support structs.");
}

class EqualtoNullSafe final : public exec::VectorFunction {
 public:
  bool isDefaultNullBehavior() const override {
    return false;
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    exec::DecodedArgs decodedArgs(rows, args, context);

    DecodedVector* decoded0 = decodedArgs.at(0);
    DecodedVector* decoded1 = decodedArgs.at(1);
    context.ensureWritable(rows, BOOLEAN(), result);
    auto* flatResult = result->asFlatVector<bool>();
    flatResult->mutableRawValues<int64_t>();

    VELOX_DYNAMIC_TYPE_DISPATCH(
        applyTyped,
        args[0]->typeKind(),
        rows,
        args,
        decoded0,
        decoded1,
        context,
        flatResult);
  }
};

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

std::shared_ptr<exec::VectorFunction> makeEqualNullSafe(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& /*config*/) {
  static const auto kEqualtoNullSafeFunction =
      std::make_shared<EqualtoNullSafe>();
  return kEqualtoNullSafeFunction;
}
} // namespace facebook::velox::functions::sparksql
