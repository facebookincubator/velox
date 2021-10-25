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

#include <utility>

#include "velox/expression/VectorFunction.h"
#include "velox/functions/lib/ArrayBuilder.h"

namespace facebook::velox::functions::sparksql {
namespace {

template <TypeKind kind>
void applyTyped(
    const SelectivityVector& rows,
    std::vector<VectorPtr>& args,
    const TypePtr& /* outputType */,
    exec::EvalCtx* context,
    VectorPtr* result) {
  using T = typename TypeTraits<kind>::NativeType;
  exec::LocalDecodedVector input0(context, *args[0], rows);
  exec::LocalDecodedVector input1(context, *args[1], rows);
  DecodedVector* decoded0 = input0.get();
  DecodedVector* decoded1 = input1.get();
  BaseVector::ensureWritable(rows, BOOLEAN(), args[0]->pool(), result);
  FlatVector<bool>* flatResult = (*result)->asFlatVector<bool>();
  flatResult->clearNulls(rows);
  flatResult->mutableRawValues<int64_t>();
  if (!args[0]->mayHaveNulls() && !args[1]->mayHaveNulls()) {
    // When there is no nulls, it reduces to normal equality function
    rows.applyToSelected([&](vector_size_t i) {
      flatResult->set(i, decoded0->valueAt<T>(i) == decoded1->valueAt<T>(i));
    });
  } else {
    // (isnull(a) AND isnull(b)) || (a == b)
    auto rawNulls0 = args[0]->flatRawNulls(rows);
    auto rawNulls1 = args[1]->flatRawNulls(rows);
    rows.applyToSelected([&](vector_size_t i) {
      auto isNull0 = bits::isBitNull(rawNulls0, i);
      auto isNull1 = bits::isBitNull(rawNulls1, i);
      flatResult->set(
          i,
          (isNull0 && isNull1) ||
              (decoded0->valueAt<T>(i) == decoded1->valueAt<T>(i)));
    });
  }
}

template <>
void applyTyped<TypeKind::ARRAY>(
    const SelectivityVector& /* rows */,
    std::vector<VectorPtr>& /* args */,
    const TypePtr& /* outputType */,
    exec::EvalCtx* /* context */,
    VectorPtr* /* result */) {
  VELOX_CHECK(false, "Bad type for EqualtoNullSafe");
}

template <>
void applyTyped<TypeKind::MAP>(
    const SelectivityVector& /* rows */,
    std::vector<VectorPtr>& /* args */,
    const TypePtr& /* outputType */,
    exec::EvalCtx* /* context */,
    VectorPtr* /* result */) {
  VELOX_CHECK(false, "Bad type for EqualtoNullSafe");
}

template <>
void applyTyped<TypeKind::ROW>(
    const SelectivityVector& /* rows */,
    std::vector<VectorPtr>& /* args */,
    const TypePtr& /* outputType */,
    exec::EvalCtx* /* context */,
    VectorPtr* /* result */) {
  VELOX_CHECK(false, "Bad type for EqualtoNullSafe");
}

class EqualtoNullSafe final : public exec::VectorFunction {
 public:
  bool isDefaultNullBehavior() const override {
    return false;
  }

  explicit EqualtoNullSafe() {}

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx* context,
      VectorPtr* result) const override {
    VELOX_DYNAMIC_TYPE_DISPATCH(
        applyTyped,
        args[0]->typeKind(),
        rows,
        args,
        outputType,
        context,
        result);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .typeVariable("T")
                .returnType("boolean")
                .argumentType("T")
                .argumentType("T")
                .build()};
  }
};

} // namespace

VELOX_DECLARE_VECTOR_FUNCTION(
    udf_equalto_nullsafe,
    EqualtoNullSafe::signatures(),
    std::make_unique<EqualtoNullSafe>());

} // namespace facebook::velox::functions::sparksql
